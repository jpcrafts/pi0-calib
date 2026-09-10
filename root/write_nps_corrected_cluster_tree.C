// Clone Hao production t_prod and add the frozen Hao-start calibration to every cluster.
// Original branches are never overwritten. Exact production-aligned seed sidecars are required
// by default because a cluster centroid is not a reliable seed-block identifier.
//
// Usage:
//   root -l -b -q 'write_hao_frozen_production_tree.C("in.root","package","out.root",4300,0,10000,"members.tsv")'

#include <TFile.h>
#include <TKey.h>
#include <TObject.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int kNCols = 30;
constexpr int kNRows = 36;
constexpr double kBlockSizeCm = 2.15;
constexpr double kXMinCm = -0.5 * kNCols * kBlockSizeCm;
constexpr double kYMinCm = -0.5 * kNRows * kBlockSizeCm;

using Row = std::map<std::string, std::string>;

std::vector<std::string> split(const std::string &text, char delimiter) {
  std::vector<std::string> values;
  std::stringstream stream(text);
  std::string value;
  while (std::getline(stream, value, delimiter)) values.push_back(value);
  return values;
}

std::vector<Row> read_tsv(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty TSV " + path);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  const auto fields = split(line, '\t');
  std::vector<Row> rows;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto values = split(line, '\t');
    Row row;
    for (size_t i = 0; i < fields.size() && i < values.size(); ++i) row[fields[i]] = values[i];
    rows.push_back(std::move(row));
  }
  return rows;
}

bool file_exists(const std::string &path) {
  std::ifstream input(path);
  return input.good();
}

double as_double(const Row &row, const std::string &field, double fallback = 0.0) {
  const auto it = row.find(field);
  if (it == row.end() || it->second.empty()) return fallback;
  return std::stod(it->second);
}

int as_int(const Row &row, const std::string &field, int fallback = 0) {
  return static_cast<int>(std::lround(as_double(row, field, fallback)));
}

struct FrozenPackage {
  std::map<std::string, std::vector<std::pair<double, double>>> log_curve;
  std::map<int, std::string> run_period;
  std::map<int, double> run_scale;
  std::map<int, double> seed_scale;
  std::string kinematic;
  std::string target;
  std::string approval_status = "unreviewed";
  std::string lowe_scope = "global";
  std::string lowe_profile = "hard_window";
  std::string lowe_shape_profile = "none";
  double lowe_min = 0.6;
  double lowe_max = 0.8;
  double lowe_full_until = 0.8;
  double lowe_unity_at = 0.8;
  std::map<std::string, double> lowe_scale;
  double lowe_damping = 0.75;
  double lowe_tilt_coefficient = 0.0;
  double lowe_tilt_min = 0.6;
  double lowe_tilt_pivot = 0.7;
  double lowe_tilt_linear_end = 0.8;
  double lowe_tilt_unity_at = 0.9;
};

FrozenPackage load_package(const std::string &directory) {
  FrozenPackage package;
  for (const auto &row : read_tsv(directory + "/package_metadata.tsv")) {
    const std::string key = row.at("key");
    const std::string value = row.at("value");
    if (key == "kinematic") package.kinematic = value;
    else if (key == "target") package.target = value;
    else if (key == "approval_status") package.approval_status = value;
    else if (key == "lowe_damping") package.lowe_damping = std::stod(value);
    else if (key == "lowe_scope") package.lowe_scope = value;
    else if (key == "lowe_profile") package.lowe_profile = value;
    else if (key == "lowe_shape_profile") package.lowe_shape_profile = value;
    else if (key == "lowe_full_until") package.lowe_full_until = std::stod(value);
    else if (key == "lowe_unity_at") package.lowe_unity_at = std::stod(value);
  }
  for (const auto &row : read_tsv(directory + "/run_period_lut.tsv")) {
    const int run = as_int(row, "run", -1);
    const std::string period = row.at("period");
    if (run <= 0 || period.empty()) throw std::runtime_error("invalid run-period row");
    const auto result = package.run_period.emplace(run, period);
    if (!result.second && result.first->second != period)
      throw std::runtime_error("conflicting run-period assignment for run " + std::to_string(run));
  }
  for (const auto &row : read_tsv(directory + "/period_photon_curve.tsv")) {
    package.log_curve[row.at("period")].push_back(
        {as_double(row, "energy_gev"), as_double(row, "log_energy_scale")});
  }
  for (auto &item : package.log_curve) std::sort(item.second.begin(), item.second.end());
  for (const auto &row : read_tsv(directory + "/run_scalar_lut.tsv")) {
    if (as_int(row, "fit_ok") == 1)
      package.run_scale[as_int(row, "run")] = as_double(row, "applied_energy_scale", 1.0);
  }
  for (const auto &row : read_tsv(directory + "/seed_block_scale.tsv")) {
    if (as_int(row, "support_pass") == 1)
      package.seed_scale[as_int(row, "seed_block")] = as_double(row, "applied_energy_scale", 1.0);
  }
  for (const auto &row : read_tsv(directory + "/lowe_photon_scale.tsv")) {
    if (row.at("fold") != "full_sample" ||
        std::fabs(as_double(row, "damping") - package.lowe_damping) > 1.0e-12) continue;
    package.lowe_min = as_double(row, "emin");
    package.lowe_max = as_double(row, "emax");
    const auto period_it = row.find("period");
    const std::string key = package.lowe_scope == "period" && period_it != row.end()
                                ? period_it->second
                                : "global";
    if (key.empty() || !package.lowe_scale.emplace(
                            key, as_double(row, "applied_photon_energy_scale", 1.0)).second)
      throw std::runtime_error("duplicate or empty low-E scale key");
  }
  const std::string tilt_path = directory + "/lowe_shape_tilt.tsv";
  if (file_exists(tilt_path)) {
    const auto rows = read_tsv(tilt_path);
    if (rows.size() != 1 || rows.front().at("fold") != "full_sample")
      throw std::runtime_error("low-E shape tilt must contain one full_sample row");
    const auto &row = rows.front();
    package.lowe_tilt_coefficient = as_double(row, "applied_log_scale_coefficient");
    package.lowe_tilt_min = as_double(row, "energy_min_gev", 0.6);
    package.lowe_tilt_pivot = as_double(row, "pivot_energy_gev", 0.7);
    package.lowe_tilt_linear_end = as_double(row, "linear_end_gev", 0.8);
    package.lowe_tilt_unity_at = as_double(row, "unity_at_gev", 0.9);
    if (!std::isfinite(package.lowe_tilt_coefficient) ||
        !(package.lowe_tilt_min < package.lowe_tilt_pivot &&
          package.lowe_tilt_pivot < package.lowe_tilt_linear_end &&
          package.lowe_tilt_linear_end < package.lowe_tilt_unity_at))
      throw std::runtime_error("invalid low-E shape tilt energy ordering");
  }
  std::set<std::string> assigned_periods;
  for (const auto &item : package.run_period) assigned_periods.insert(item.second);
  bool lowe_complete = package.lowe_scope == "global"
                           ? package.lowe_scale.size() == 1 && package.lowe_scale.count("global")
                           : package.lowe_scope == "period" &&
                                 package.lowe_scale.size() == assigned_periods.size();
  if (package.lowe_scope == "period") {
    for (const auto &period : assigned_periods)
      lowe_complete = lowe_complete && package.lowe_scale.count(period);
  }
  if (package.kinematic.empty() || package.target.empty() || package.run_period.empty() ||
      package.log_curve.empty() || package.run_scale.empty() || package.seed_scale.empty() || !lowe_complete)
    throw std::runtime_error("frozen package is incomplete or ambiguous: " + directory);
  for (const auto &item : package.run_scale) {
    const auto period = package.run_period.find(item.first);
    if (period == package.run_period.end())
      throw std::runtime_error("run scale lacks period assignment for run " + std::to_string(item.first));
    if (!package.log_curve.count(period->second))
      throw std::runtime_error("run period lacks curve: " + period->second);
    if (package.lowe_scope == "period" && !package.lowe_scale.count(period->second))
      throw std::runtime_error("run period lacks low-E scale: " + period->second);
  }
  return package;
}

double curve_scale(const FrozenPackage &package, int run, double energy) {
  const auto period = package.run_period.find(run);
  if (period == package.run_period.end()) return std::numeric_limits<double>::quiet_NaN();
  auto it = package.log_curve.find(period->second);
  if (it == package.log_curve.end() || it->second.empty()) return std::numeric_limits<double>::quiet_NaN();
  const auto &points = it->second;
  if (energy <= points.front().first) return std::exp(points.front().second);
  if (energy >= points.back().first) return std::exp(points.back().second);
  for (size_t i = 1; i < points.size(); ++i) {
    if (energy <= points[i].first) {
      const double fraction = (energy - points[i - 1].first) / (points[i].first - points[i - 1].first);
      return std::exp((1.0 - fraction) * points[i - 1].second + fraction * points[i].second);
    }
  }
  return std::numeric_limits<double>::quiet_NaN();
}

double lowe_tilt_basis(const FrozenPackage &package, double energy) {
  if (energy < package.lowe_tilt_min || energy >= package.lowe_tilt_unity_at) return 0.0;
  if (energy <= package.lowe_tilt_linear_end)
    return (package.lowe_tilt_pivot - energy) /
           (package.lowe_tilt_pivot - package.lowe_tilt_min);
  const double width = package.lowe_tilt_unity_at - package.lowe_tilt_linear_end;
  const double t = (energy - package.lowe_tilt_linear_end) / width;
  const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  const double h10 = t * t * t - 2.0 * t * t + t;
  const double start_value =
      (package.lowe_tilt_pivot - package.lowe_tilt_linear_end) /
      (package.lowe_tilt_pivot - package.lowe_tilt_min);
  const double start_derivative = -1.0 /
      (package.lowe_tilt_pivot - package.lowe_tilt_min);
  return h00 * start_value + h10 * width * start_derivative;
}

double lowe_scale(const FrozenPackage &package, int run, double energy) {
  double profile_scale = 1.0;
  if (energy >= package.lowe_min && energy < package.lowe_max) {
    const std::string key = package.lowe_scope == "period" ? package.run_period.at(run) : "global";
    const double amplitude = package.lowe_scale.at(key);
    if (package.lowe_profile == "hard_window") profile_scale = amplitude;
    else if (package.lowe_profile == "smoothstep") {
      if (energy <= package.lowe_full_until) profile_scale = amplitude;
      else if (energy < package.lowe_unity_at) {
        const double t = (energy - package.lowe_full_until) /
                         (package.lowe_unity_at - package.lowe_full_until);
        const double weight = 1.0 - (3.0 * t * t - 2.0 * t * t * t);
        profile_scale = std::pow(amplitude, weight);
      }
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  return profile_scale * std::exp(
      package.lowe_tilt_coefficient * lowe_tilt_basis(package, energy));
}

// Read only member_index==0 rows. This avoids materializing a multi-GB sidecar as maps of strings.
std::map<std::pair<Long64_t, int>, int> load_sidecar_seeds(const std::string &path) {
  std::map<std::pair<Long64_t, int>, int> seeds;
  if (path.empty()) return seeds;
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open seed sidecar " + path);
  std::string line;
  std::getline(input, line);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  const auto header = split(line, '\t');
  std::map<std::string, size_t> index;
  for (size_t i = 0; i < header.size(); ++i) index[header[i]] = i;
  for (const std::string field : {"t_entry", "prod_cluster_slot", "seed_block", "member_index"})
    if (!index.count(field)) throw std::runtime_error("sidecar lacks " + field + ": " + path);
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto value = split(line, '\t');
    if (value.size() < header.size() || std::stoi(value[index["member_index"]]) != 0) continue;
    const Long64_t entry = std::stoll(value[index["t_entry"]]);
    const int slot = std::stoi(value[index["prod_cluster_slot"]]);
    const int seed = std::stoi(value[index["seed_block"]]);
    const auto key = std::make_pair(entry, slot);
    auto result = seeds.emplace(key, seed);
    if (!result.second && result.first->second != seed)
      throw std::runtime_error("conflicting sidecar seed assignment");
  }
  return seeds;
}

int centroid_block(double x, double y) {
  int col = static_cast<int>(std::floor((x - kXMinCm) / kBlockSizeCm));
  int row = static_cast<int>(std::floor((y - kYMinCm) / kBlockSizeCm));
  col = std::max(0, std::min(kNCols - 1, col));
  row = std::max(0, std::min(kNRows - 1, row));
  return row * kNCols + col;
}

}  // namespace

void write_nps_corrected_cluster_tree(
    const char *input_root,
    const char *package_dir,
    const char *output_root,
    int run_override = -1,
    int segment = -1,
    Long64_t max_entries = -1,
    const char *sidecar_tsv = "",
    bool require_exact_seed = true,
    bool identity_on_missing_seed = false,
    bool copy_auxiliary_objects = true,
    bool allow_unvalidated = false) {
  if (std::string(input_root) == std::string(output_root)) {
    std::cerr << "input and output ROOT paths must differ\n";
    return;
  }
  FrozenPackage package;
  std::map<std::pair<Long64_t, int>, int> sidecar_seed;
  try {
    package = load_package(package_dir);
    if (package.approval_status != "validated" && !allow_unvalidated)
      throw std::runtime_error(
          "package approval_status is " + package.approval_status);
    sidecar_seed = load_sidecar_seeds(sidecar_tsv ? sidecar_tsv : "");
  } catch (const std::exception &error) {
    std::cerr << "setup failure: " << error.what() << "\n";
    return;
  }

  TFile input(input_root, "READ");
  TTree *source = input.IsOpen() ? dynamic_cast<TTree *>(input.Get("t_prod")) : nullptr;
  if (!source) { std::cerr << "cannot read t_prod from " << input_root << "\n"; return; }
  for (const std::string &branch : {"g.runnum", "NPS.prod.clusE", "NPS.prod.clusX", "NPS.prod.clusY"}) {
    if (!source->GetBranch(branch.c_str())) {
      std::cerr << "input t_prod lacks required branch " << branch << "\n";
      return;
    }
  }
  for (const std::string &branch : {
           "NPS_haoFinal_clusE", "NPS_haoFinal_scale", "NPS_haoFinal_curve_scale",
           "NPS_haoFinal_run_scale", "NPS_haoFinal_seed_scale", "NPS_haoFinal_lowe_scale",
           "NPS_haoFinal_seed_block", "NPS_haoFinal_seed_source"}) {
    if (source->GetBranch(branch.c_str())) {
      std::cerr << "input already contains calibration branch " << branch << "\n";
      return;
    }
  }

  double runnum = 0.0;
  std::vector<double> *clusE = nullptr, *clusX = nullptr, *clusY = nullptr;
  source->SetBranchAddress("g.runnum", &runnum);
  source->SetBranchAddress("NPS.prod.clusE", &clusE);
  source->SetBranchAddress("NPS.prod.clusX", &clusX);
  source->SetBranchAddress("NPS.prod.clusY", &clusY);

  const std::string output_path(output_root);
  const auto slash = output_path.find_last_of('/');
  if (slash != std::string::npos) gSystem->mkdir(output_path.substr(0, slash).c_str(), true);
  TFile output(output_root, "RECREATE");
  if (copy_auxiliary_objects) {
    std::set<std::string> copied;
    TIter next_key(input.GetListOfKeys());
    while (auto *key = dynamic_cast<TKey *>(next_key())) {
      const std::string name = key->GetName();
      if (name == "t_prod" || !copied.insert(name).second) continue;
      TObject *object = key->ReadObj();
      if (!object) {
        std::cerr << "failed to copy input object " << name << "\n";
        output.Close(); input.Close(); gSystem->Unlink(output_root); return;
      }
      output.cd();
      object->Write(name.c_str());
      delete object;
    }
  }
  output.cd();
  TTree *tree = source->CloneTree(0);

  int run = 0;
  Long64_t missing_seed_total = 0, exact_seed_total = 0, centroid_seed_total = 0;
  Long64_t identity_seed_total = 0, failed_scale_total = 0;
  Long64_t invalid_cluster_vectors_total = 0, missing_run_scale_total = 0;
  std::vector<double> final_e, total_scale, curve_component, run_component, seed_component, lowe_component;
  std::vector<int> seed_block, seed_source;
  tree->Branch("NPS_haoFinal_clusE", &final_e);
  tree->Branch("NPS_haoFinal_scale", &total_scale);
  tree->Branch("NPS_haoFinal_curve_scale", &curve_component);
  tree->Branch("NPS_haoFinal_run_scale", &run_component);
  tree->Branch("NPS_haoFinal_seed_scale", &seed_component);
  tree->Branch("NPS_haoFinal_lowe_scale", &lowe_component);
  tree->Branch("NPS_haoFinal_seed_block", &seed_block);
  tree->Branch("NPS_haoFinal_seed_source", &seed_source);

  const Long64_t entries_in = source->GetEntries();
  const Long64_t entries = max_entries > 0 ? std::min(max_entries, entries_in) : entries_in;
  for (Long64_t entry = 0; entry < entries; ++entry) {
    source->GetEntry(entry);
    run = run_override > 0 ? run_override : static_cast<int>(std::lround(runnum));
    const size_t clusters = clusE ? clusE->size() : 0;
    final_e = clusE ? *clusE : std::vector<double>{};
    total_scale.assign(clusters, 1.0); curve_component.assign(clusters, 1.0);
    run_component.assign(clusters, 1.0); seed_component.assign(clusters, 1.0);
    lowe_component.assign(clusters, 1.0); seed_block.assign(clusters, -1); seed_source.assign(clusters, 0);

    if (!clusE || !clusX || !clusY || clusX->size() != clusters || clusY->size() != clusters) {
      ++invalid_cluster_vectors_total; tree->Fill(); continue;
    }
    const auto run_it = package.run_scale.find(run);
    if (run_it == package.run_scale.end() || !package.run_period.count(run)) {
      ++missing_run_scale_total; tree->Fill(); continue;
    }

    for (size_t cluster = 0; cluster < clusters; ++cluster) {
      const auto sidecar_it = sidecar_seed.find({entry, static_cast<int>(cluster)});
      int seed = -1;
      if (sidecar_it != sidecar_seed.end()) {
        seed = sidecar_it->second; seed_source[cluster] = 1; ++exact_seed_total;
      } else if (identity_on_missing_seed) {
        // Preserve the cluster and apply all non-spatial layers. A missing exact
        // seed is safer with unity spatial scale than with a guessed block.
        seed_source[cluster] = 3; ++identity_seed_total;
      } else if (!require_exact_seed) {
        seed = centroid_block(clusX->at(cluster), clusY->at(cluster));
        seed_source[cluster] = 2; ++centroid_seed_total;
      } else {
        ++missing_seed_total; continue;
      }
      seed_block[cluster] = seed;
      const double c = curve_scale(package, run, clusE->at(cluster));
      const double r = run_it->second;
      const auto seed_it = package.seed_scale.find(seed);
      const double s = seed_it == package.seed_scale.end() ? 1.0 : seed_it->second;
      const double l = lowe_scale(package, run, clusE->at(cluster));
      const double scale = c * r * s * l;
      if (!std::isfinite(scale) || scale <= 0.0) { ++failed_scale_total; continue; }
      curve_component[cluster] = c; run_component[cluster] = r; seed_component[cluster] = s;
      lowe_component[cluster] = l; total_scale[cluster] = scale; final_e[cluster] = clusE->at(cluster) * scale;
    }
    tree->Fill();
  }

  tree->Write();
  TTree metadata("hao_final_metadata", "frozen Hao-start calibration writer metadata");
  std::string metadata_input(input_root), metadata_package(package_dir), metadata_sidecar(sidecar_tsv ? sidecar_tsv : "");
  std::string metadata_kinematic = package.kinematic, metadata_target = package.target;
  std::string metadata_approval_status = package.approval_status;
  std::string metadata_lowe_profile = package.lowe_profile;
  std::string metadata_lowe_shape_profile = package.lowe_shape_profile;
  double metadata_lowe_tilt_coefficient = package.lowe_tilt_coefficient;
  Long64_t metadata_entries_in = entries_in, metadata_entries_written = entries;
  int metadata_run = run_override;
  int metadata_segment = segment;
  int metadata_require_exact = require_exact_seed ? 1 : 0;
  int metadata_copied_auxiliary = copy_auxiliary_objects ? 1 : 0;
  metadata.Branch("input_root", &metadata_input); metadata.Branch("package_dir", &metadata_package);
  metadata.Branch("sidecar_tsv", &metadata_sidecar);
  metadata.Branch("kinematic", &metadata_kinematic);
  metadata.Branch("target", &metadata_target);
  metadata.Branch("approval_status", &metadata_approval_status);
  metadata.Branch("lowe_profile", &metadata_lowe_profile);
  metadata.Branch("lowe_shape_profile", &metadata_lowe_shape_profile);
  metadata.Branch("lowe_tilt_coefficient", &metadata_lowe_tilt_coefficient,
                  "lowe_tilt_coefficient/D");
  metadata.Branch("entries_in", &metadata_entries_in, "entries_in/L");
  metadata.Branch("entries_written", &metadata_entries_written, "entries_written/L");
  metadata.Branch("run", &metadata_run, "run/I");
  metadata.Branch("segment", &metadata_segment, "segment/I");
  metadata.Branch("exact_seed_clusters", &exact_seed_total, "exact_seed_clusters/L");
  metadata.Branch("centroid_seed_clusters", &centroid_seed_total, "centroid_seed_clusters/L");
  metadata.Branch("identity_seed_clusters", &identity_seed_total, "identity_seed_clusters/L");
  metadata.Branch("missing_seed_clusters", &missing_seed_total, "missing_seed_clusters/L");
  metadata.Branch("failed_scale_clusters", &failed_scale_total, "failed_scale_clusters/L");
  metadata.Branch("invalid_cluster_vector_events", &invalid_cluster_vectors_total, "invalid_cluster_vector_events/L");
  metadata.Branch("missing_run_scale_events", &missing_run_scale_total, "missing_run_scale_events/L");
  metadata.Branch("require_exact_seed", &metadata_require_exact, "require_exact_seed/I");
  metadata.Branch("copied_auxiliary_objects", &metadata_copied_auxiliary,
                  "copied_auxiliary_objects/I");
  metadata.Fill(); metadata.Write(); output.Close(); input.Close();

  std::cout << "entries_in " << entries_in << "\nentries_written " << entries
            << "\nexact_seed_clusters " << exact_seed_total
            << "\ncentroid_seed_clusters " << centroid_seed_total << "\nmissing_seed_clusters " << missing_seed_total
            << "\nidentity_seed_clusters " << identity_seed_total
            << "\nfailed_scale_clusters " << failed_scale_total
            << "\ninvalid_cluster_vector_events " << invalid_cluster_vectors_total
            << "\nmissing_run_scale_events " << missing_run_scale_total << "\n";
  if ((require_exact_seed && missing_seed_total > 0) || failed_scale_total > 0 ||
      invalid_cluster_vectors_total > 0 || missing_run_scale_total > 0) {
    std::cerr << "calibration coverage failed; deleting incomplete output " << output_root << "\n";
    gSystem->Unlink(output_root);
  }
}
