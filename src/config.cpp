#include "config.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pcz {

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

void Config::load_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open input file: " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  source_ = path;
  parse_text(ss.str());
}

void Config::parse_text(const std::string& text) {
  std::istringstream in(text);
  std::string line, section;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    auto c = line.find_first_of("#");
    if (c != std::string::npos) line = line.substr(0, c);
    line = trim(line);
    if (line.empty()) continue;
    if (line.front() == '[') {
      auto e = line.find(']');
      if (e == std::string::npos) throw std::runtime_error("bad section header at line " + std::to_string(lineno));
      section = trim(line.substr(1, e - 1));
      continue;
    }
    auto eq = line.find('=');
    if (eq == std::string::npos) throw std::runtime_error("expected key = value at line " + std::to_string(lineno));
    std::string key = trim(line.substr(0, eq));
    std::string val = trim(line.substr(eq + 1));
    kv_[section.empty() ? key : section + "." + key] = val;
  }
}

void Config::set(const std::string& key, const std::string& value) { kv_[key] = value; }

bool Config::has(const std::string& key) const { return kv_.count(key) > 0; }

std::string Config::get_string(const std::string& key, const std::string& def) const {
  auto it = kv_.find(key);
  used_.insert(key);
  return it == kv_.end() ? def : it->second;
}

double Config::get_double(const std::string& key, double def) const {
  auto it = kv_.find(key);
  used_.insert(key);
  if (it == kv_.end()) return def;
  try {
    return std::stod(it->second);
  } catch (...) {
    throw std::runtime_error("key '" + key + "' expects a number, got '" + it->second + "'");
  }
}

int Config::get_int(const std::string& key, int def) const {
  auto it = kv_.find(key);
  used_.insert(key);
  if (it == kv_.end()) return def;
  try {
    return std::stoi(it->second);
  } catch (...) {
    throw std::runtime_error("key '" + key + "' expects an integer, got '" + it->second + "'");
  }
}

bool Config::get_bool(const std::string& key, bool def) const {
  auto it = kv_.find(key);
  used_.insert(key);
  if (it == kv_.end()) return def;
  std::string v = it->second;
  std::transform(v.begin(), v.end(), v.begin(), ::tolower);
  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  throw std::runtime_error("key '" + key + "' expects a boolean, got '" + it->second + "'");
}

static std::vector<double> parse_list(const std::string& s) {
  std::vector<double> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = trim(tok);
    if (!tok.empty()) out.push_back(std::stod(tok));
  }
  return out;
}

std::vector<double> Config::get_list(const std::string& key, const std::vector<double>& def) const {
  auto it = kv_.find(key);
  used_.insert(key);
  if (it == kv_.end()) return def;
  if (trim(it->second).empty() || trim(it->second) == "none") return {};
  return parse_list(it->second);
}

std::vector<std::vector<double>> Config::get_groups(const std::string& key) const {
  auto it = kv_.find(key);
  used_.insert(key);
  std::vector<std::vector<double>> out;
  if (it == kv_.end()) return out;
  std::stringstream ss(it->second);
  std::string grp;
  while (std::getline(ss, grp, ';')) {
    grp = trim(grp);
    if (!grp.empty() && grp != "none") out.push_back(parse_list(grp));
  }
  return out;
}

std::vector<std::string> Config::unused_keys() const {
  std::vector<std::string> out;
  for (auto& [k, v] : kv_)
    if (!used_.count(k)) out.push_back(k);
  return out;
}

std::string Config::dump() const {
  std::ostringstream o;
  for (auto& [k, v] : kv_) o << k << " = " << v << "\n";
  return o.str();
}

}  // namespace pcz
