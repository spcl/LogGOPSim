#ifndef TRACE_SCHEMA_HPP
#define TRACE_SCHEMA_HPP

#include <string>
#include <vector>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

/*
 * Self-describing GOAL binary format (GOALX).
 *
 * The header embeds a schema: a list of fields with a kind, a bit width and a
 * bit offset inside a fixed-size, bit-packed node record. Writer (txt2bin) and
 * reader (LogGOPSim) both go through this file, so the layout is defined once.
 *
 * File layout (little-endian):
 *
 *   0  u64 magic (GOALX_MAGIC)
 *   8  u16 version · u8 endian · u8 flags · u32 header_len
 *  16  u64 schema_hash
 *  24  u32 num_ranks · u32 max_cpu · u32 max_nic
 *  36  u16 num_fields · u16 record_bits · u16 record_bytes · u16 num_meta
 *  44  field[i]: u16 kind · u16 bit_offset · u8 width · u8 flags · u64 default · u8 name_len · name
 *      meta[i]:  u16 klen · key · u16 vlen · value
 *  header_len: jump table, num_ranks x { u64 start, u64 end } (absolute file offsets)
 *
 *  rank block: u32 num_nodes · u32 num_roots · u32 roots[num_roots]
 *              · u32 depcnt_init[num_nodes] · records[num_nodes * record_bytes]
 *              · u32 appendix[num_edges] · GOALX_PADDING zero bytes
 *
 * The padding guarantees that an unaligned 64-bit load at any field's byte
 * offset stays inside the file.
 */

#define GOALX_MAGIC          0x0031584C414F4723ULL  /* "#GOALX1\0" */
#define GOALX_VERSION        1
#define GOALX_ENDIAN_LITTLE  1
#define GOALX_PADDING        8

#define GOALX_OFF_MAX_CPU    28
#define GOALX_OFF_MAX_NIC    32
#define GOALX_OFF_FIELDS     44

/* field kinds the simulator knows about; everything else is FK_CUSTOM */
enum FieldKind {
	FK_TYPE = 0,
	FK_PEER,
	FK_SIZE,
	FK_TAG,
	FK_CPU,
	FK_NIC,
	FK_NDEP,    // number of nodes depending on this node's completion
	FK_DEPIX,   // start of that list in the appendix
	FK_NSDEP,   // number of nodes depending on this node's start
	FK_SDEPIX,  // start of that list (optional, defaults to depix + ndep)
	FK_NUM_BUILTIN,
	FK_CUSTOM = 0xFFFF
};

/* decode all-ones-of-width as ANY (~0u), used for peer and tag */
#define FF_ANY_SENTINEL 0x1

static const char* const builtin_field_names[FK_NUM_BUILTIN] = {
	"type", "peer", "size", "tag", "cpu", "nic", "ndep", "depix", "nsdep", "sdepix"
};

/* default widths of a GOALX schema; a width of 0 means "not in the schema" */
static const uint8_t builtin_field_default_widths[FK_NUM_BUILTIN] = {
	2, 32, 64, 32, 8, 8, 32, 32, 32, 0
};

static inline bool host_is_little_endian() {
	uint16_t x = 1;
	return *((uint8_t*) &x) == 1;
}

static inline uint64_t width_mask(uint8_t width) {
	return width >= 64 ? ~0ULL : ((1ULL << width) - 1);
}

static inline bool check_fits(uint64_t value, uint8_t width) {
	return value <= width_mask(width);
}

struct FieldDesc {
	uint16_t    kind;
	uint16_t    bit_offset;
	uint8_t     width;
	uint8_t     flags;
	uint64_t    dflt;
	std::string name;
};

/* precomputed decoding information for one field: no loop, no switch at decode time */
struct Accessor {
	uint32_t byte_off;
	uint8_t  shift;
	bool     split;          // the field crosses into the 9th byte of the load
	bool     present;
	bool     any_sentinel;
	uint64_t mask;
	uint64_t dflt;

	Accessor() : byte_off(0), shift(0), split(false), present(false), any_sentinel(false), mask(0), dflt(0) {}

	explicit Accessor(const FieldDesc* f) {
		if (f == NULL) {
			*this = Accessor();
			return;
		}
		byte_off     = f->bit_offset / 8;
		shift        = f->bit_offset % 8;
		split        = shift + f->width > 64;
		present      = true;
		any_sentinel = (f->flags & FF_ANY_SENTINEL) != 0;
		mask         = width_mask(f->width);
		dflt         = f->dflt;
	}

	inline uint64_t get(const char* rec) const {
		if (!present) return dflt;
		uint64_t w;
		memcpy(&w, rec + byte_off, sizeof(w));
		uint64_t v = w >> shift;
		if (split) v |= ((uint64_t) (uint8_t) rec[byte_off + 8]) << (64 - shift);
		v &= mask;
		if (any_sentinel && v == mask) return 0xFFFFFFFFULL;
		return v;
	}
};

/* writes value into the record at bit_offset; rec needs 8 bytes of slack after the record */
static inline void put_bits(char* rec, uint16_t bit_offset, uint8_t width, uint64_t value) {
	uint32_t byte_off = bit_offset / 8;
	uint8_t shift = bit_offset % 8;
	uint64_t mask = width_mask(width);
	uint64_t w;

	value &= mask;
	memcpy(&w, rec + byte_off, sizeof(w));
	w = (w & ~(mask << shift)) | (value << shift);
	memcpy(rec + byte_off, &w, sizeof(w));
	if (shift + width > 64) {
		uint8_t hi_mask = (uint8_t) (mask >> (64 - shift));
		uint8_t b = (uint8_t) rec[byte_off + 8];
		b = (b & ~hi_mask) | (uint8_t) (value >> (64 - shift));
		rec[byte_off + 8] = (char) b;
	}
}

static inline uint64_t fnv1a(const char* data, size_t len, uint64_t h = 0xcbf29ce484222325ULL) {
	for (size_t i = 0; i < len; i++) {
		h ^= (uint8_t) data[i];
		h *= 0x100000001b3ULL;
	}
	return h;
}

class Schema {

	public:

	std::vector<FieldDesc> fields;
	std::vector<std::pair<std::string, std::string> > meta;
	uint16_t record_bits;
	uint16_t record_bytes;
	bool     legacy;

	Schema() : record_bits(0), record_bytes(0), legacy(false) {}

	const FieldDesc* find(const std::string& name) const {
		for (size_t i = 0; i < fields.size(); i++) {
			if (fields[i].name == name) return &fields[i];
		}
		return NULL;
	}

	const FieldDesc* builtin(FieldKind kind) const {
		for (size_t i = 0; i < fields.size(); i++) {
			if (fields[i].kind == kind) return &fields[i];
		}
		return NULL;
	}

	static int builtin_kind(const std::string& name) {
		for (int k = 0; k < FK_NUM_BUILTIN; k++) {
			if (name == builtin_field_names[k]) return k;
		}
		return FK_CUSTOM;
	}

	/*
	 * The fixed 39-byte record of MAGIC_COOKIE (4223) files, depcnt at byte 0.
	 * sDepIx (byte 35) is left out: txt2bin always writes depIx + nDep there,
	 * and a 64-bit load at byte 35 would read past the last record.
	 */
	static Schema legacy_layout() {
		static const uint16_t byte_offsets[FK_SDEPIX] = { 4, 5, 9, 17, 21, 22, 23, 27, 31 };
		static const uint8_t widths[FK_SDEPIX] = { 8, 32, 64, 32, 8, 8, 32, 32, 32 };
		Schema s;
		for (int k = 0; k < FK_SDEPIX; k++) {
			FieldDesc f;
			f.kind = k;
			f.bit_offset = byte_offsets[k] * 8;
			f.width = widths[k];
			f.flags = (k == FK_PEER || k == FK_TAG) ? FF_ANY_SENTINEL : 0;
			f.dflt = 0;
			f.name = builtin_field_names[k];
			s.fields.push_back(f);
		}
		s.record_bytes = 39;
		s.record_bits = 39 * 8;
		s.legacy = true;
		return s;
	}

	/*
	 * Builds a GOALX schema from the default widths, overridden or extended by
	 * a spec like "peer:17,tag:24,size:40,gpu_id:6=0". Builtin fields come
	 * first in their fixed order, custom fields follow in spec order.
	 */
	static bool from_spec(const char* spec, Schema* out, std::string* err) {
		uint8_t widths[FK_NUM_BUILTIN];
		uint64_t defaults[FK_NUM_BUILTIN];
		std::vector<FieldDesc> customs;

		memcpy(widths, builtin_field_default_widths, sizeof(widths));
		memset(defaults, 0, sizeof(defaults));

		std::string s = spec ? spec : "";
		size_t pos = 0;
		while (pos < s.size()) {
			size_t end = s.find(',', pos);
			if (end == std::string::npos) end = s.size();
			std::string item = s.substr(pos, end - pos);
			pos = end + 1;
			if (item.empty()) continue;

			size_t colon = item.find(':');
			if (colon == std::string::npos || colon == 0) {
				*err = "schema entry '" + item + "' must look like name:width[=default]";
				return false;
			}
			std::string name = item.substr(0, colon);
			std::string rest = item.substr(colon + 1);
			uint64_t dflt = 0;
			size_t eq = rest.find('=');
			if (eq != std::string::npos) {
				dflt = strtoull(rest.c_str() + eq + 1, NULL, 10);
				rest = rest.substr(0, eq);
			}
			int width = atoi(rest.c_str());
			if (width < 0 || width > 64) {
				*err = "field '" + name + "' has width " + rest + ", must be between 0 and 64";
				return false;
			}

			if (width > 0 && !check_fits(dflt, (uint8_t) width)) {
				*err = "default of field '" + name + "' does not fit in " + rest + " bits";
				return false;
			}

			int kind = builtin_kind(name);
			if (kind != FK_CUSTOM) {
				widths[kind] = (uint8_t) width;
				defaults[kind] = dflt;
			} else {
				if (width == 0) {
					*err = "custom field '" + name + "' needs a width of at least 1";
					return false;
				}
				FieldDesc f;
				f.kind = FK_CUSTOM;
				f.width = (uint8_t) width;
				f.flags = 0;
				f.dflt = dflt;
				f.name = name;
				customs.push_back(f);
			}
		}

		Schema r;
		uint32_t bit = 0;
		for (int k = 0; k < FK_NUM_BUILTIN; k++) {
			if (widths[k] == 0) continue;
			FieldDesc f;
			f.kind = k;
			f.bit_offset = bit;
			f.width = widths[k];
			f.flags = (k == FK_PEER || k == FK_TAG) ? FF_ANY_SENTINEL : 0;
			f.dflt = defaults[k];
			f.name = builtin_field_names[k];
			r.fields.push_back(f);
			bit += f.width;
		}
		for (size_t i = 0; i < customs.size(); i++) {
			if (r.find(customs[i].name) != NULL) {
				*err = "field '" + customs[i].name + "' is declared twice";
				return false;
			}
			customs[i].bit_offset = bit;
			r.fields.push_back(customs[i]);
			bit += customs[i].width;
		}
		if (bit > 0xFFFF) {
			*err = "record is too large";
			return false;
		}
		r.record_bits = bit;
		r.record_bytes = (bit + 7) / 8;
		*out = r;
		return true;
	}

	/* the bytes that define the record layout; their hash identifies the schema */
	void serialize_fields(std::vector<char>& buf) const {
		for (size_t i = 0; i < fields.size(); i++) {
			const FieldDesc& f = fields[i];
			put(buf, f.kind);
			put(buf, f.bit_offset);
			put(buf, f.width);
			put(buf, f.flags);
			put(buf, f.dflt);
			put(buf, (uint8_t) f.name.size());
			buf.insert(buf.end(), f.name.begin(), f.name.end());
		}
	}

	uint64_t hash() const {
		std::vector<char> buf;
		serialize_fields(buf);
		put(buf, record_bits);
		put(buf, record_bytes);
		return fnv1a(buf.data(), buf.size());
	}

	/* everything before the jump table */
	void serialize_header(std::vector<char>& buf, uint32_t num_ranks, uint32_t max_cpu, uint32_t max_nic) const {
		buf.clear();
		put(buf, (uint64_t) GOALX_MAGIC);
		put(buf, (uint16_t) GOALX_VERSION);
		put(buf, (uint8_t) GOALX_ENDIAN_LITTLE);
		put(buf, (uint8_t) 0);
		put(buf, (uint32_t) 0); // header_len, patched below
		put(buf, hash());
		put(buf, num_ranks);
		put(buf, max_cpu);
		put(buf, max_nic);
		put(buf, (uint16_t) fields.size());
		put(buf, record_bits);
		put(buf, record_bytes);
		put(buf, (uint16_t) meta.size());
		serialize_fields(buf);
		for (size_t i = 0; i < meta.size(); i++) {
			put(buf, (uint16_t) meta[i].first.size());
			buf.insert(buf.end(), meta[i].first.begin(), meta[i].first.end());
			put(buf, (uint16_t) meta[i].second.size());
			buf.insert(buf.end(), meta[i].second.begin(), meta[i].second.end());
		}
		uint32_t header_len = buf.size();
		memcpy(&buf[12], &header_len, sizeof(header_len));
	}

	/* parses everything before the jump table and checks it for consistency */
	static bool parse_header(const char* hdr, size_t len, Schema* out, uint32_t* num_ranks,
	                         uint32_t* max_cpu, uint32_t* max_nic, uint32_t* header_len, std::string* err) {
		size_t pos = 0;
		uint64_t magic, hash;
		uint16_t version, num_fields, num_meta;
		uint8_t endian, flags;
		Schema s;

		if (len < GOALX_OFF_FIELDS) {
			*err = "file is too short for a GOALX header";
			return false;
		}
		get(hdr, pos, magic);
		get(hdr, pos, version);
		get(hdr, pos, endian);
		get(hdr, pos, flags);
		get(hdr, pos, *header_len);
		get(hdr, pos, hash);
		get(hdr, pos, *num_ranks);
		get(hdr, pos, *max_cpu);
		get(hdr, pos, *max_nic);
		get(hdr, pos, num_fields);
		get(hdr, pos, s.record_bits);
		get(hdr, pos, s.record_bytes);
		get(hdr, pos, num_meta);
		(void) flags;

		if (magic != GOALX_MAGIC) {
			*err = "not a GOALX file";
			return false;
		}
		if (version != GOALX_VERSION) {
			*err = "unsupported GOALX version " + std::to_string(version) + " (this build reads version " + std::to_string(GOALX_VERSION) + ")";
			return false;
		}
		if (endian != GOALX_ENDIAN_LITTLE || !host_is_little_endian()) {
			*err = "byte order of the trace does not match this host";
			return false;
		}
		if (*header_len > len || *header_len < GOALX_OFF_FIELDS) {
			*err = "header length " + std::to_string(*header_len) + " is out of range";
			return false;
		}

		len = *header_len;
		for (uint16_t i = 0; i < num_fields; i++) {
			FieldDesc f;
			uint8_t name_len;
			if (pos + 15 > len) { *err = "truncated field descriptor"; return false; }
			get(hdr, pos, f.kind);
			get(hdr, pos, f.bit_offset);
			get(hdr, pos, f.width);
			get(hdr, pos, f.flags);
			get(hdr, pos, f.dflt);
			get(hdr, pos, name_len);
			if (pos + name_len > len) { *err = "truncated field name"; return false; }
			f.name.assign(hdr + pos, name_len);
			pos += name_len;
			if (f.width < 1 || f.width > 64 || f.bit_offset + f.width > s.record_bits) {
				*err = "field '" + f.name + "' does not fit into the record";
				return false;
			}
			if (f.kind != FK_CUSTOM && (f.kind >= FK_NUM_BUILTIN || s.builtin((FieldKind) f.kind) != NULL)) {
				*err = "field '" + f.name + "' has an invalid or duplicate kind";
				return false;
			}
			s.fields.push_back(f);
		}
		for (uint16_t i = 0; i < num_meta; i++) {
			uint16_t klen, vlen;
			if (pos + 2 > len) { *err = "truncated metadata"; return false; }
			get(hdr, pos, klen);
			if (pos + klen + 2 > len) { *err = "truncated metadata"; return false; }
			std::string key(hdr + pos, klen);
			pos += klen;
			get(hdr, pos, vlen);
			if (pos + vlen > len) { *err = "truncated metadata"; return false; }
			s.meta.push_back(std::make_pair(key, std::string(hdr + pos, vlen)));
			pos += vlen;
		}
		if (s.record_bytes != (s.record_bits + 7) / 8) {
			*err = "record size does not match record bits";
			return false;
		}
		if (s.hash() != hash) {
			*err = "schema hash mismatch, the header is corrupt";
			return false;
		}
		*out = s;
		return true;
	}

	void print(FILE* fd, uint32_t num_ranks, uint32_t max_cpu, uint32_t max_nic) const {
		fprintf(fd, "format: %s\n", legacy ? "legacy (magic 4223, fixed 39-byte records)" : "GOALX v1");
		if (!legacy) fprintf(fd, "schema hash: %016" PRIx64 "\n", hash());
		fprintf(fd, "ranks: %u  max cpu: %u  max nic: %u\n", num_ranks, max_cpu, max_nic);
		fprintf(fd, "record: %u bits, %u bytes\n", record_bits, record_bytes);
		fprintf(fd, "fields:\n");
		for (size_t i = 0; i < fields.size(); i++) {
			const FieldDesc& f = fields[i];
			fprintf(fd, "  %-10s bits %3u..%3u  width %2u%s%s", f.name.c_str(), f.bit_offset,
			        f.bit_offset + f.width - 1, f.width, f.kind == FK_CUSTOM ? "  custom" : "",
			        (f.flags & FF_ANY_SENTINEL) ? "  (all ones = ANY)" : "");
			if (f.dflt != 0) fprintf(fd, "  default %" PRIu64, f.dflt);
			fprintf(fd, "\n");
		}
		for (size_t i = 0; i < meta.size(); i++) {
			fprintf(fd, "meta: %s=%s\n", meta[i].first.c_str(), meta[i].second.c_str());
		}
	}

	private:

	template <typename T>
	static void put(std::vector<char>& buf, T v) {
		const char* p = (const char*) &v;
		buf.insert(buf.end(), p, p + sizeof(T));
	}

	template <typename T>
	static void get(const char* buf, size_t& pos, T& v) {
		memcpy(&v, buf + pos, sizeof(T));
		pos += sizeof(T);
	}
};

/* what a consumer of the trace needs: a field name and a minimum width */
struct Requirement {
	std::string name;
	uint8_t     min_width;
};

static inline bool parse_requirements(const char* spec, std::vector<Requirement>* out, std::string* err) {
	std::string s = spec ? spec : "";
	size_t pos = 0;
	while (pos < s.size()) {
		size_t end = s.find(',', pos);
		if (end == std::string::npos) end = s.size();
		std::string item = s.substr(pos, end - pos);
		pos = end + 1;
		if (item.empty()) continue;
		Requirement r;
		size_t colon = item.find(':');
		r.name = item.substr(0, colon);
		r.min_width = colon == std::string::npos ? 1 : (uint8_t) atoi(item.c_str() + colon + 1);
		if (r.name.empty()) {
			*err = "requirement '" + item + "' has no field name";
			return false;
		}
		out->push_back(r);
	}
	return true;
}

/*
 * Checks that the trace provides what the simulator needs. The simulator
 * itself keeps peer/tag in 32 bits and cpu/nic in 8 bits, so wider values
 * are refused here instead of being truncated later.
 */
static inline bool validate_schema(const Schema& s, const std::vector<Requirement>& extra,
                                   uint32_t num_ranks, uint32_t max_cpu, uint32_t max_nic, std::string* err) {
	static const FieldKind required[] = { FK_TYPE, FK_PEER, FK_SIZE, FK_TAG, FK_NDEP, FK_DEPIX, FK_NSDEP };
	for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
		if (s.builtin(required[i]) == NULL) {
			*err = std::string("does not provide required field '") + builtin_field_names[required[i]] + "'";
			return false;
		}
	}
	for (size_t i = 0; i < extra.size(); i++) {
		const FieldDesc* f = s.find(extra[i].name);
		if (f == NULL) {
			*err = "does not provide required field '" + extra[i].name + "'";
			return false;
		}
		if (f->width < extra[i].min_width) {
			*err = "field '" + extra[i].name + "' is " + std::to_string(f->width) + " bits wide, at least " +
			       std::to_string(extra[i].min_width) + " are required";
			return false;
		}
	}

	if (s.builtin(FK_TYPE)->width < 2) {
		*err = "field 'type' needs at least 2 bits";
		return false;
	}
	const FieldKind max32[] = { FK_PEER, FK_TAG, FK_NDEP, FK_DEPIX, FK_NSDEP, FK_SDEPIX };
	for (size_t i = 0; i < sizeof(max32) / sizeof(max32[0]); i++) {
		const FieldDesc* f = s.builtin(max32[i]);
		if (f != NULL && f->width > 32) {
			*err = "field '" + f->name + "' is " + std::to_string(f->width) + " bits wide, this simulator supports at most 32";
			return false;
		}
	}
	const FieldDesc* peer = s.builtin(FK_PEER);
	uint64_t peer_limit = width_mask(peer->width); // all ones is reserved for ANY_SOURCE
	if (num_ranks > 0 && (uint64_t) (num_ranks - 1) >= peer_limit) {
		*err = "field 'peer' (" + std::to_string(peer->width) + " bits) cannot address " + std::to_string(num_ranks) + " ranks";
		return false;
	}
	if (max_cpu > 255 || max_nic > 255) {
		*err = "trace uses cpu " + std::to_string(max_cpu) + " / nic " + std::to_string(max_nic) + ", this simulator supports at most 255";
		return false;
	}
	return true;
}

#endif
