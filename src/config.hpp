// Minimal INI-style configuration: "[section]" headers and "key = value" lines.
// Keys are stored as "section.key". Command-line overrides use the same form.
#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>

namespace pcz {

class Config {
public:
  Config() = default;
  void load_file(const std::string& path);
  void parse_text(const std::string& text);
  void set(const std::string& key, const std::string& value);  // "section.key"

  bool has(const std::string& key) const;
  std::string get_string(const std::string& key, const std::string& def) const;
  double get_double(const std::string& key, double def) const;
  int get_int(const std::string& key, int def) const;
  bool get_bool(const std::string& key, bool def) const;
  std::vector<double> get_list(const std::string& key, const std::vector<double>& def) const;
  // Groups separated by ';', each a comma list, e.g. "0,0,1e-3,2e-4,0.5; ..."
  std::vector<std::vector<double>> get_groups(const std::string& key) const;

  std::vector<std::string> unused_keys() const;
  std::string dump() const;  // all key/values (for provenance in output dir)
  const std::string& source() const { return source_; }

private:
  std::map<std::string, std::string> kv_;
  mutable std::set<std::string> used_;
  std::string source_;
};

std::string trim(const std::string& s);

}  // namespace pcz
