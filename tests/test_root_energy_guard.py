"""Optional integration test: run under the ROOT environment, no farm inputs needed."""
from __future__ import annotations

import array
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from nps_pi0_calibration import FrozenCalibration

try:
    import ROOT
except ImportError:
    ROOT = None

BASE = Path(__file__).resolve().parents[1]
PACKAGE = BASE / "packages/x60_4_lh2_v7_guarded"


@unittest.skipUnless(ROOT is not None and shutil.which("root"), "requires PyROOT and ROOT executable")
class RootEnergyGuardTest(unittest.TestCase):
    def test_all_cluster_cpp_python_parity(self) -> None:
        energies = (0.7, 1.2, 2.0, 2.000001, 2.1, 2.25, 2.4, 2.499999, 2.5, 2.500001, 4.0, 6.0)
        calibration = FrozenCalibration.load(PACKAGE)
        macros = [BASE / "root/write_nps_corrected_cluster_tree.C"]
        # Also exercise the site writer without creating a standalone dependency on it.
        if os.environ.get("PI0_SITE_WRITER"):
            macros.append(Path(os.environ["PI0_SITE_WRITER"]))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "input.root"
            output = ROOT.TFile(str(source), "RECREATE")
            tree = ROOT.TTree("t_prod", "mixed-energy events without Pi0 selection")
            run = array.array("d", [4253.0])
            tree.Branch("g.runnum", run, "run/D")
            e, x, y = (ROOT.std.vector("double")() for _ in range(3))
            for branch, vector in (("clusE", e), ("clusX", x), ("clusY", y)):
                tree.Branch("NPS.prod." + branch, vector)
            for energy in energies:
                e.push_back(energy); x.push_back(0.0); y.push_back(0.0)
            tree.Fill()
            e.clear(); x.clear(); y.clear()
            tree.Fill()  # Empty-cluster event must also survive.
            tree.Write()
            hist = ROOT.TH1D("original_qa", "unchanged", 10, 0, 10)
            hist.Fill(3)
            hist.Write()
            output.Close()
            sidecar = tmp / "members.tsv"
            sidecar.write_text("t_entry\tprod_cluster_slot\tseed_block\tmember_index\n" +
                               "".join(f"0\t{i}\t608\t0\n" for i in range(len(energies))))
            for number, macro in enumerate(macros):
                with self.subTest(writer=macro.name):
                    destination = tmp / f"out{number}.root"
                    tail = ",true,true" if number == 0 else ",true"
                    call = (f'{macro}("{source}","{PACKAGE}","{destination}",'
                            f'4253,0,-1,"{sidecar}",true,false{tail})')
                    result = subprocess.run(["root", "-l", "-b", "-q", call],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue(destination.is_file(), result.stdout + result.stderr)
                    data = ROOT.TFile.Open(str(destination))
                    actual = data.Get("t_prod")
                    self.assertEqual(actual.GetEntries(), 2)
                    actual.GetEntry(0)
                    self.assertEqual(list(getattr(actual, "NPS.prod.clusE")), list(energies))
                    for index, energy in enumerate(energies):
                        expected = calibration.correct(4253, energy, 608)
                        components = []
                        for field in ("curve_scale", "run_scale", "seed_scale", "lowe_scale"):
                            value = getattr(actual, "NPS_haoFinal_" + field)[index]
                            components.append(value)
                            self.assertAlmostEqual(value, getattr(expected, field), places=13)
                        corrected = getattr(actual, "NPS_haoFinal_clusE")[index]
                        scale = getattr(actual, "NPS_haoFinal_scale")[index]
                        self.assertAlmostEqual(corrected, expected.corrected_energy_gev, places=13)
                        self.assertEqual(scale, math.prod(components))
                        if energy >= 2.5:
                            self.assertEqual(corrected, energy)
                            self.assertEqual(components, [1.0] * 4)
                    actual.GetEntry(1)
                    self.assertEqual(len(getattr(actual, "NPS_haoFinal_clusE")), 0)
                    meta = data.Get("hao_final_metadata")
                    meta.GetEntry(0)
                    self.assertEqual(meta.high_energy_identity_clusters, 4)
                    self.assertEqual(meta.high_energy_taper_clusters, 5)
                    if number == 0:
                        self.assertEqual(data.Get("original_qa").GetEntries(), 1)
                    data.Close()


if __name__ == "__main__":
    unittest.main()
