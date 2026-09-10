#include <TChain.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
double inv_mass_gev(
    const std::vector<double>& px,
    const std::vector<double>& py,
    const std::vector<double>& pz,
    const std::vector<double>& ene,
    int i,
    int j) {
  const double e = ene.at(i) + ene.at(j);
  const double x = px.at(i) + px.at(j);
  const double y = py.at(i) + py.at(j);
  const double z = pz.at(i) + pz.at(j);
  const double m2 = e * e - x * x - y * y - z * z;
  return m2 > 0.0 ? std::sqrt(m2) : std::numeric_limits<double>::quiet_NaN();
}
}  // namespace

void extract_production_exact2_pairs(
    int run = 4300,
    int segment = 0,
    const char* input_file = "nps_production_4300_0_wf_calib.root",
    const char* output_tsv = "production_exact2_pairs.tsv",
    double min_e_gev = 0.6,
    double max_e_gev = 2.5,
    double timing_half_window_ns = 1.0,
    double timing_center_ns = -0.176158,
    Long64_t max_entries = -1) {
  TChain ch("t_prod");
  ch.Add(input_file);

  double g_runnum = 0.0;
  double g_evnum = 0.0;
  std::vector<double>* clus_e = nullptr;
  std::vector<int>* clus_size = nullptr;
  std::vector<double>* clus_t = nullptr;
  std::vector<double>* clus_x = nullptr;
  std::vector<double>* clus_y = nullptr;
  std::vector<double>* px = nullptr;
  std::vector<double>* py = nullptr;
  std::vector<double>* pz = nullptr;
  std::vector<double>* ene = nullptr;

  ch.SetBranchAddress("g.runnum", &g_runnum);
  ch.SetBranchAddress("g.evnum", &g_evnum);
  ch.SetBranchAddress("NPS.prod.clusE", &clus_e);
  ch.SetBranchAddress("NPS.prod.clusSize", &clus_size);
  ch.SetBranchAddress("NPS.prod.clusT", &clus_t);
  ch.SetBranchAddress("NPS.prod.clusX", &clus_x);
  ch.SetBranchAddress("NPS.prod.clusY", &clus_y);
  ch.SetBranchAddress("NPS.prod.trk.px", &px);
  ch.SetBranchAddress("NPS.prod.trk.py", &py);
  ch.SetBranchAddress("NPS.prod.trk.pz", &pz);
  ch.SetBranchAddress("NPS.prod.trk.ene", &ene);

  const std::string out_path(output_tsv);
  const auto slash = out_path.find_last_of('/');
  if (slash != std::string::npos) gSystem->mkdir(out_path.substr(0, slash).c_str(), true);
  std::ofstream out(out_path);
  out << "t_entry\trun\tsegment\tevent\tpair_i\tpair_j\tpair_m\t"
      << "e1\te2\tx1\ty1\tx2\ty2\tt1\tt2\tdt12\tsize1\tsize2\n";

  Long64_t selected_events = 0;
  const Long64_t nentries = ch.GetEntries();
  const Long64_t entries_to_scan = (max_entries > 0 && max_entries < nentries) ? max_entries : nentries;
  for (Long64_t ie = 0; ie < entries_to_scan; ++ie) {
    ch.GetEntry(ie);
    if (!clus_e || !clus_size || !clus_t || !clus_x || !clus_y || !px || !py || !pz || !ene) continue;
    const size_t n = std::min({clus_e->size(), clus_size->size(), clus_t->size(), clus_x->size(),
                               clus_y->size(), px->size(), py->size(), pz->size(), ene->size()});
    std::vector<int> selected;
    for (size_t i = 0; i < n; ++i) {
      if (clus_e->at(i) < min_e_gev || clus_e->at(i) > max_e_gev) continue;
      if (std::abs(clus_t->at(i) - timing_center_ns) > timing_half_window_ns) continue;
      selected.push_back(static_cast<int>(i));
    }
    if (selected.size() != 2) continue;
    const int i = selected[0];
    const int j = selected[1];
    const double m = inv_mass_gev(*px, *py, *pz, *ene, i, j);
    if (!std::isfinite(m)) continue;
    ++selected_events;
    out << ie << '\t'
        << static_cast<int>(std::lround(g_runnum)) << '\t'
        << segment << '\t'
        << static_cast<long long>(std::llround(g_evnum)) << '\t'
        << i << '\t' << j << '\t' << m << '\t'
        << clus_e->at(i) << '\t' << clus_e->at(j) << '\t'
        << clus_x->at(i) << '\t' << clus_y->at(i) << '\t'
        << clus_x->at(j) << '\t' << clus_y->at(j) << '\t'
        << clus_t->at(i) << '\t' << clus_t->at(j) << '\t'
        << std::abs(clus_t->at(i) - clus_t->at(j)) << '\t'
        << clus_size->at(i) << '\t' << clus_size->at(j) << '\n';
  }

  std::cout << "input_file " << input_file << "\n";
  std::cout << "entries " << nentries << "\n";
  std::cout << "entries_scanned " << entries_to_scan << "\n";
  std::cout << "selected_events_exact2 " << selected_events << "\n";
  std::cout << "output " << output_tsv << "\n";
}
