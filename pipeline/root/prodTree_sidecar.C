#include "TSystem.h"
#include "TMatrix.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TFile.h"
#include "TCutG.h"
#include "TChain.h"
#include "TCanvas.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TLorentzVector.h"
#include "TNtuple.h"
#include "TLegend.h"
#include "TLine.h"
#include "TRandom3.h"
#include "TStyle.h"
#include <iostream>
#include <sstream>
#include <vector>
// #include "TDVCSGlobal.h"
#include "TTreeIndex.h"
#include "TChainIndex.h"
#include "TCaloEvent.h"
#include "TCaloGeometry.h"
#include "TCaloBase.h"
#include "TDVCSEvent.h"
#include <fstream>
#include <iomanip>

using namespace std;

Int_t ndecay = 5000; // number of decays for pi0 contamination simulation

// Functions to convert the numbering scheme of NPS
Int_t bnConv_OldToNew(int ibn_old);
Int_t bnConv_NewToOld(int ibn_new);
bool readVectorFile(const TString &path, std::vector<Double_t> &vals, int expected);
bool readIntVectorFile(const TString &path, std::vector<Int_t> &vals, int expected);
bool readScalarFile(const TString &path, Double_t &val);

void prodTree_sidecar(int run_number, int iseg, const char *inputDirArg=".", const char *outputDirArg=".", const char *sidecarDirArg=".", Int_t max_events=-1, const char *dbHostArg="jmysql.jlab.org", const char *dbDirArg="", Bool_t enable_pi0sub=false)
{   
    TH1::SetDefaultSumw2();

    Double_t m_p = 0.938272013; // proton mass [GeV]
    Double_t m_pi0 = 0.1349766; // pi0 mass [GeV]

    // Kinematics dependent variables that are not in DB________________________________________
    Double_t clusTrsH = 0.2; // GeV
    Double_t pi0_sigm = 0.0046; // GeV, for pi0 mass window cut of pi0 contamination subtraction
    
    // Input rootfiles__________________________________________________________
    TString dataDir = inputDirArg;
    TString filename = Form("nps_production_%d_%d_wf.root", run_number, iseg);

    TFile *infile = TFile::Open(Form("%s/%s", dataDir.Data(), filename.Data()));
    if(!infile || infile->IsZombie()){
        std::cerr<<Form("Error: can not open %s/%s", dataDir.Data(), filename.Data())<<endl;
        return;
    }

    // Output rootfiles__________________________________________________________
    TString outputDir = outputDirArg;
    TString outfilename = Form("nps_production_%d_%d_wf_calib.root", run_number, iseg);
    TString sidecarDir = sidecarDirArg;
    gSystem->mkdir(sidecarDir, true);
    TString sidecarName = Form("%s/nps_production_%d_%d_wf_calib.production_members.tsv", sidecarDir.Data(), run_number, iseg);
    std::ofstream sidecar(sidecarName.Data());
    sidecar << std::setprecision(16);
    sidecar << "t_entry\twf_entry\trun\tsegment\tevent\ttime_window_index\ttime_window_center_ns\t"
            << "prod_cluster_slot\tsource_cluster_index\tcluster_e\tcluster_size\tcluster_x\tcluster_y\tcluster_t\t"
            << "seed_block\tseed_block_old\tseed_row\tseed_col\tseed_energy\t"
            << "member_index\tblock\tblock_old\trow\tcol\tmember_energy\tmember_time\tmember_fraction\n";
    if(!sidecar){
        std::cerr << "Error creating sidecar file " << sidecarName << "\n";
        return;
    }

    TFile *outfile = new TFile(Form("%s/%s", outputDir.Data(), outfilename.Data()), "recreate");

    if (!outfile){ // Check if the output file was created successfully
        std::cerr << "Error creating output file.\n";
        return;
    }

    // T tree__________________________________________________________
    TTree *t_T = (TTree*)infile->Get("T");
    if (!t_T){
        std::cerr << "Error: no T tree found in the file.\n";
        // infile->Close();
        return;
    }
    
    Double_t runNb_T;
    Double_t evtNb_T;

    Double_t H_react_ok;
    Double_t H_react_x;
    Double_t H_react_y;
    Double_t H_react_z;

    Double_t H_gtr_ok;
    Double_t H_gtr_dp;
    Double_t H_gtr_ph;
    Double_t H_gtr_th;
    Double_t H_gtr_px;
    Double_t H_gtr_py;
    Double_t H_gtr_pz;

    Double_t H_dc_ntrack;

    Double_t H_hod_beta;
    Double_t H_hod_goodscinhit;
    Double_t H_hod_goodstarttime;
    Double_t H_hod_betanotrack;

    Double_t H_cer_npeSum;

    Double_t H_cal_etotnorm;
    Double_t H_cal_etracknorm;
    Double_t H_cal_etottracknorm;
    Double_t H_cal_eprtracknorm;

    Double_t T_hms_hEDTM_tdcTimeRaw;
    Double_t H_1MHz_scaler;
    Double_t H_BCM4A_scaler;
    Double_t H_BCM4A_scalerCharge;
    Double_t H_BCM4A_scalerCurrent;
    Double_t H_BCM4A_scalerChargeCut;
    Double_t H_BCM4A_Hel_scalerCharge;
    Double_t H_BCM4A_Hel_scalerCurrent;
    Double_t H_BCM4A_Hel_scaler;
    Double_t T_helicity_hel;

    // Disabla all and turn on the branches we need
    t_T->SetBranchStatus("*", false);

    t_T->SetBranchStatus("g.runnum", true);
    t_T->SetBranchStatus("g.evnum", true);

    t_T->SetBranchStatus("H.react.ok", true);
    t_T->SetBranchStatus("H.react.x", true);
    t_T->SetBranchStatus("H.react.y", true);
    t_T->SetBranchStatus("H.react.z", true);

    t_T->SetBranchStatus("H.gtr.ok", true);
    t_T->SetBranchStatus("H.gtr.px", true);
    t_T->SetBranchStatus("H.gtr.py", true);
    t_T->SetBranchStatus("H.gtr.pz", true);
    t_T->SetBranchStatus("H.gtr.dp", true);
    t_T->SetBranchStatus("H.gtr.ph", true);
    t_T->SetBranchStatus("H.gtr.th", true);

    t_T->SetBranchStatus("H.dc.ntrack", true);

    t_T->SetBranchStatus("H.hod.beta", true);
    t_T->SetBranchStatus("H.hod.goodscinhit", true);
    t_T->SetBranchStatus("H.hod.goodstarttime", true);
    t_T->SetBranchStatus("H.hod.betanotrack", true);

    t_T->SetBranchStatus("H.cer.npeSum", true);

    t_T->SetBranchStatus("H.cal.etracknorm", true);
    t_T->SetBranchStatus("H.cal.etotnorm", true);
    t_T->SetBranchStatus("H.cal.etottracknorm", true);
    t_T->SetBranchStatus("H.cal.eprtracknorm", true);
    
    t_T->SetBranchStatus("T.hms.hEDTM_tdcTimeRaw", true);
    t_T->SetBranchStatus("H.1MHz.scaler", true);
    t_T->SetBranchStatus("H.BCM4A.scaler", true);
    t_T->SetBranchStatus("H.BCM4A.scalerCharge", true);
    t_T->SetBranchStatus("H.BCM4A.scalerCurrent", true);
    t_T->SetBranchStatus("H.BCM4A.scalerChargeCut", true);
    t_T->SetBranchStatus("H.BCM4A_Hel.scalerCharge", true);
    t_T->SetBranchStatus("H.BCM4A_Hel.scalerCurrent", true);
    t_T->SetBranchStatus("H.BCM4A_Hel.scaler", true);
    t_T->SetBranchStatus("T.helicity.hel", true);

    //Event Level variables
    t_T->SetBranchAddress("g.runnum", &runNb_T);
    t_T->SetBranchAddress("g.evnum", &evtNb_T); // Global event number of T tree
    // HMS information
    t_T->SetBranchAddress("H.react.ok", &H_react_ok);
    t_T->SetBranchAddress("H.react.x", &H_react_x);
    t_T->SetBranchAddress("H.react.y", &H_react_y);
    t_T->SetBranchAddress("H.react.z", &H_react_z);

    t_T->SetBranchAddress("H.gtr.ok", &H_gtr_ok);
    t_T->SetBranchAddress("H.gtr.px", &H_gtr_px);
    t_T->SetBranchAddress("H.gtr.py", &H_gtr_py);
    t_T->SetBranchAddress("H.gtr.pz", &H_gtr_pz);
    t_T->SetBranchAddress("H.gtr.dp", &H_gtr_dp);
    t_T->SetBranchAddress("H.gtr.ph", &H_gtr_ph);
    t_T->SetBranchAddress("H.gtr.th", &H_gtr_th);

    t_T->SetBranchAddress("H.dc.ntrack", &H_dc_ntrack);
    
    t_T->SetBranchAddress("H.hod.goodscinhit", &H_hod_goodscinhit);
    t_T->SetBranchAddress("H.hod.goodstarttime", &H_hod_goodstarttime);
    t_T->SetBranchAddress("H.hod.betanotrack", &H_hod_betanotrack);
    t_T->SetBranchAddress("H.hod.beta", &H_hod_beta);

    t_T->SetBranchAddress("H.cer.npeSum", &H_cer_npeSum);
    
    t_T->SetBranchAddress("H.cal.etracknorm", &H_cal_etracknorm);
    t_T->SetBranchAddress("H.cal.etotnorm", &H_cal_etotnorm);
    t_T->SetBranchAddress("H.cal.etottracknorm", &H_cal_etottracknorm);
    t_T->SetBranchAddress("H.cal.eprtracknorm", &H_cal_eprtracknorm);

    t_T->SetBranchAddress("T.hms.hEDTM_tdcTimeRaw", &T_hms_hEDTM_tdcTimeRaw);
    t_T->SetBranchAddress("H.1MHz.scaler", &H_1MHz_scaler);
    t_T->SetBranchAddress("H.BCM4A.scaler", &H_BCM4A_scaler);
    t_T->SetBranchAddress("H.BCM4A.scalerCharge", &H_BCM4A_scalerCharge);
    t_T->SetBranchAddress("H.BCM4A.scalerCurrent", &H_BCM4A_scalerCurrent);
    t_T->SetBranchAddress("H.BCM4A.scalerChargeCut", &H_BCM4A_scalerChargeCut);
    t_T->SetBranchAddress("H.BCM4A_Hel.scalerCharge", &H_BCM4A_Hel_scalerCharge);
    t_T->SetBranchAddress("H.BCM4A_Hel.scalerCurrent", &H_BCM4A_Hel_scalerCurrent);
    t_T->SetBranchAddress("H.BCM4A_Hel.scaler", &H_BCM4A_Hel_scaler);
    t_T->SetBranchAddress("T.helicity.hel", &T_helicity_hel);

    // TChain of waveform tree___________________________________________________________
    TTree *t_wf = (TTree*)infile->Get("WF");
    if (!t_wf) {
        std::cerr << "Error: no WF tree found in the file.\n";
        return;
    }

    Double_t evtNb_wf;
    vector<Double_t> *chi2 = nullptr;
    vector<Int_t> *wfnpulse = nullptr;
    vector<Double_t> *wfampl = nullptr;
    vector<Double_t> *wftime = nullptr;
    t_wf->SetBranchAddress("evt", &evtNb_wf); // Global event number of wf tree
    t_wf->SetBranchAddress("chi2", &chi2); // Number of pulses found by TSpectrum in this block
    t_wf->SetBranchAddress("wfnpulse", &wfnpulse); // Number of pulses found by TSpectrum in this block 
    t_wf->SetBranchAddress("wfampl", &wfampl); // Waveform amplitude of each fitted pulse (mV)
    t_wf->SetBranchAddress("wftime", &wftime); // Pulse time of each fitted pulse per block (ns)

    // Use the TVirtualIndex to access the entry corresponding to the on in T tree
    TVirtualIndex *vIdx = t_wf->GetTreeIndex();
    TTreeIndex *idx = dynamic_cast<TTreeIndex *>(vIdx);
    if (!idx)
    {
        std::cerr << "Error: no TTreeIndex found in the output tree.\n";
        infile->Close();
        return;
    }

    // Get the index array
    Long64_t *indexArray = idx->GetIndex();
    Int_t nevt_wf = t_wf->GetEntries();
    Int_t nevt_T = t_T->GetEntries();
    Double_t lastevent = 0.;

    if (nevt_wf != nevt_T){ // Check the number of events in both trees
        std::cerr << "Error: number of events in WF tree (" << nevt_wf << ") does not match number of events in T tree (" << nevt_T << ").\n";
        return;
    }

    // Connect to the database and get kinematics variables________________________________________________
    const char *dvcsLib = gSystem->Getenv("NPS_DVCS_LIB");
    if (!dvcsLib || TString(dvcsLib).IsNull()) dvcsLib = "libDVCS.so";
    if (gSystem->Load(dvcsLib) < 0) {
        std::cerr << "Error loading DVCS library " << dvcsLib
                  << "; set NPS_DVCS_LIB or initialize replay environment\n";
        return;
    }
    Double_t Beam_energy = 0.; // Beam energy in GeV
    Double_t HMS_mom = 0.; // HMS central momentum in GeV/c
    Double_t HMS_angle = 0.; // HMS angle in rad
    Double_t Target_amu = 0.; // Target amu
    Double_t NPS_dist = 0.; // NPS distance in cm
    Double_t NPS_angle = 0.; // NPS angle in rad
    std::vector<Double_t> timingOffsetVec;
    std::vector<Double_t> coefPi0Vec;
    std::vector<Int_t> caloMaskBlockVec;
    Double_t *timingOffset = nullptr;
    Double_t *coefPi0 = nullptr;
    Int_t *caloMaskBlock = nullptr;

    TString dbDir = dbDirArg;
    if (dbDir.Length() == 0) {
        std::cerr << "Error: frozen DB snapshot directory is required\n";
        return;
    }
    {
        bool ok = true;
        ok &= readScalarFile(Form("%s/BEAM_param_Energy.value.txt", dbDir.Data()), Beam_energy);
        ok &= readScalarFile(Form("%s/SIMU_param_HMSmomentum.value.txt", dbDir.Data()), HMS_mom);
        ok &= readScalarFile(Form("%s/SIMU_param_HMSangle.value.txt", dbDir.Data()), HMS_angle);
        ok &= readScalarFile(Form("%s/TARGET_param_Amu.value.txt", dbDir.Data()), Target_amu);
        ok &= readScalarFile(Form("%s/CALO_geom_Dist.value.txt", dbDir.Data()), NPS_dist);
        ok &= readScalarFile(Form("%s/CALO_geom_Yaw.value.txt", dbDir.Data()), NPS_angle);
        ok &= readVectorFile(Form("%s/CALO_calib_TimeOffset.txt", dbDir.Data()), timingOffsetVec, 1080);
        ok &= readVectorFile(Form("%s/CALO_calib_Pi0Coef.txt", dbDir.Data()), coefPi0Vec, 1080);
        ok &= readIntVectorFile(Form("%s/CALO_flag_MaskBlock.txt", dbDir.Data()), caloMaskBlockVec, 1080);
        if (!ok)
        {
            std::cerr << "Error: failed to read production sidecar DB cache from " << dbDir << "\n";
            return;
        }
        timingOffset = timingOffsetVec.data();
        coefPi0 = coefPi0Vec.data();
        caloMaskBlock = caloMaskBlockVec.data();
    }

    cout<<"Start clustering for Run "<<run_number<<", segment "<<iseg<<endl;
    cout<<"========== Run information =========="<<endl;
    cout<<"Beam energy: "<<Beam_energy<<" GeV"<<endl;
    cout<<"HMS momentum: "<<HMS_mom<<" GeV/c"<<endl;
    cout<<"HMS angle: "<<HMS_angle*TMath::RadToDeg()<<" degrees"<<endl;
    cout<<"Target amu: "<<Target_amu<<" GeV/c^2"<<endl;
    cout<<"NPS distance: "<<NPS_dist<<" cm"<<endl;
    cout<<"NPS angle: "<<NPS_angle*TMath::RadToDeg()<<" degrees"<<endl;
    cout<<"======================================"<<endl;
    cout<<endl;
    cout<<"NPS 2x2 clustering threshold "<<clusTrsH<<" GeV"<<endl;
    cout<<endl;
    cout<<"========== NPS maske blocks =========="<<endl;
    for(int i = 0; i < 1080; i++) if(caloMaskBlock[i] == 1) cout<<i<<" ";
    cout<<endl;
    cout<<"======================================"<<endl;
    cout<<endl;
    cout<<"========== NPS pi0 calibration coefficients =========="<<endl;
    for(int i = 0; i < 1080; i++) cout<<coefPi0[i]<<" ";
    cout<<endl;
    cout<<"======================================"<<endl;
    cout<<endl;
    cout<<"========== NPS timing offsets =========="<<endl;
    for(int iblk = 0; iblk < 1080; iblk++) cout<<timingOffset[iblk]<<" ";
    cout<<endl;
    cout<<"======================================"<<endl;
    cout<<endl;

    // Initial settings for clustering__________________________________________________
    TLorentzVector beam(0, 0, Beam_energy, Beam_energy);
    TLorentzVector p0(0, 0, 0, m_p);

    TDVCSEvent *ev = new TDVCSEvent(run_number);
    TCaloEvent *caloev = new TCaloEvent(run_number);
    ev->GetGeometry()->SetCaloTheta(-1*NPS_angle); // rad Note: the angle have to be negative here
    ev->GetGeometry()->SetCaloDist(NPS_dist); // cm

    // Ontput Tree for production data___________________________________________________________
    TTree *t_prod = new TTree("t_prod", "t_prod");

    // Variables for NPS information in the tree
    Int_t nclust, nclust_acc1, nclust_coin, nclust_acc2;
    vector<double> clusE;
    vector<int> clusSize;
    vector<double> clusX;
    vector<double> clusX_corr;
    vector<double> clusY;
    vector<double> clusY_corr;
    vector<double> clusT;
    vector<double> clusDepth;
    vector<double> trkPx;
    vector<double> trkPy;
    vector<double> trkPz;
    vector<double> trkE;
    vector<double> M;
    vector<double> Mx2;

    //Event Level variables
    t_prod->Branch("g.runnum", &runNb_T);
    t_prod->Branch("g.evnum", &evtNb_T); // Global event number of T tree
    // vertex information
    t_prod->Branch("H.react.ok", &H_react_ok);
    t_prod->Branch("H.react.x", &H_react_x);
    t_prod->Branch("H.react.y", &H_react_y);
    t_prod->Branch("H.react.z", &H_react_z);
    // HMS information
    t_prod->Branch("H.gtr.ok", &H_gtr_ok);
    t_prod->Branch("H.gtr.px", &H_gtr_px);
    t_prod->Branch("H.gtr.py", &H_gtr_py);
    t_prod->Branch("H.gtr.pz", &H_gtr_pz);
    t_prod->Branch("H.gtr.dp", &H_gtr_dp);
    t_prod->Branch("H.gtr.ph", &H_gtr_ph);
    t_prod->Branch("H.gtr.th", &H_gtr_th);

    t_prod->Branch("H.dc.ntrack", &H_dc_ntrack);
    
    t_prod->Branch("H.hod.beta", &H_hod_beta);
    t_prod->Branch("H.hod.goodscinhit", &H_hod_goodscinhit);
    t_prod->Branch("H.hod.goodstarttime", &H_hod_goodstarttime);
    t_prod->Branch("H.hod.betanotrack", &H_hod_betanotrack);

    t_prod->Branch("H.cer.npeSum", &H_cer_npeSum);

    t_prod->Branch("H.cal.etotnorm", &H_cal_etotnorm);
    t_prod->Branch("H.cal.etracknorm", &H_cal_etracknorm);
    t_prod->Branch("H.cal.etottracknorm", &H_cal_etottracknorm);
    t_prod->Branch("H.cal.eprtracknorm", &H_cal_eprtracknorm);
    
    t_prod->Branch("T.helicity.hel", &T_helicity_hel);
    t_prod->Branch("T.hms.hEDTM_tdcTimeRaw", &T_hms_hEDTM_tdcTimeRaw);

    t_prod->Branch("H.1MHz.scaler", &H_1MHz_scaler);
    t_prod->Branch("H.BCM4A.scaler", &H_BCM4A_scaler);
    t_prod->Branch("H.BCM4A.scalerCharge", &H_BCM4A_scalerCharge);
    t_prod->Branch("H.BCM4A.scalerCurrent", &H_BCM4A_scalerCurrent);
    t_prod->Branch("H.BCM4A.scalerChargeCut", &H_BCM4A_scalerChargeCut);
    t_prod->Branch("H.BCM4A_Hel.scalerCharge", &H_BCM4A_Hel_scalerCharge);
    t_prod->Branch("H.BCM4A_Hel.scalerCurrent", &H_BCM4A_Hel_scalerCurrent);
    t_prod->Branch("H.BCM4A_Hel.scaler", &H_BCM4A_Hel_scaler);
    // NPS information
    t_prod->Branch("NPS.prod.nclust", &nclust, "nclust_of_all_timing_window/I");
    t_prod->Branch("NPS.prod.nclustAcc1", &nclust_acc1, "nclust_accidental_negative_timing/I");
    t_prod->Branch("NPS.prod.nclustCoin", &nclust_coin, "nclust_coin_time/I");
    t_prod->Branch("NPS.prod.nclustAcc2", &nclust_acc2, "nclust_accidental_positive_timing/I");
    t_prod->Branch("NPS.prod.clusE", &clusE);
    t_prod->Branch("NPS.prod.clusSize", &clusSize);
    t_prod->Branch("NPS.prod.clusX", &clusX);
    t_prod->Branch("NPS.prod.clusXcorr", &clusX_corr);
    t_prod->Branch("NPS.prod.clusY", &clusY);
    t_prod->Branch("NPS.prod.clusYcorr", &clusY_corr);
    t_prod->Branch("NPS.prod.clusZ", &NPS_dist);
    t_prod->Branch("NPS.prod.clusT", &clusT);
    t_prod->Branch("NPS.prod.clusDepth", &clusDepth);
    t_prod->Branch("NPS.prod.trk.px", &trkPx);
    t_prod->Branch("NPS.prod.trk.py", &trkPy);
    t_prod->Branch("NPS.prod.trk.pz", &trkPz);
    t_prod->Branch("NPS.prod.trk.ene", &trkE);
    t_prod->Branch("NPS.prod.M", &M);
    t_prod->Branch("NPS.prod.Mx2", &Mx2);

    // Ontput Tree for pi0 contamination___________________________________________________________
    TTree *t_pi0sub = new TTree("t_pi0sub", "t_pi0sub");
    //Event Level variables
    t_pi0sub->Branch("g.runnum", &runNb_T);
    t_pi0sub->Branch("g.evnum", &evtNb_T); // Global event number of T tree
    // vertex information
    t_pi0sub->Branch("H.react.ok", &H_react_ok);
    t_pi0sub->Branch("H.react.x", &H_react_x);
    t_pi0sub->Branch("H.react.y", &H_react_y);
    t_pi0sub->Branch("H.react.z", &H_react_z);
    // HMS information
    t_pi0sub->Branch("H.gtr.ok", &H_gtr_ok);
    t_pi0sub->Branch("H.gtr.px", &H_gtr_px);
    t_pi0sub->Branch("H.gtr.py", &H_gtr_py);
    t_pi0sub->Branch("H.gtr.pz", &H_gtr_pz);
    t_pi0sub->Branch("H.gtr.dp", &H_gtr_dp);
    t_pi0sub->Branch("H.gtr.ph", &H_gtr_ph);
    t_pi0sub->Branch("H.gtr.th", &H_gtr_th);
    
    t_pi0sub->Branch("H.dc.ntrack", &H_dc_ntrack);

    t_pi0sub->Branch("H.hod.beta", &H_hod_beta);
    t_pi0sub->Branch("H.hod.goodscinhit", &H_hod_goodscinhit);
    t_pi0sub->Branch("H.hod.goodstarttime", &H_hod_goodstarttime);
    t_pi0sub->Branch("H.hod.betanotrack", &H_hod_betanotrack);

    t_pi0sub->Branch("H.cer.npeSum", &H_cer_npeSum);

    t_pi0sub->Branch("H.cal.etotnorm", &H_cal_etotnorm);
    t_pi0sub->Branch("H.cal.etracknorm", &H_cal_etracknorm);
    t_pi0sub->Branch("H.cal.etottracknorm", &H_cal_etottracknorm);
    t_pi0sub->Branch("H.cal.eprtracknorm", &H_cal_eprtracknorm);
    
    t_pi0sub->Branch("T.helicity.hel", &T_helicity_hel);
    t_pi0sub->Branch("T.hms.hEDTM_tdcTimeRaw", &T_hms_hEDTM_tdcTimeRaw);

    t_pi0sub->Branch("H.1MHz.scaler", &H_1MHz_scaler);
    
    t_pi0sub->Branch("H.BCM4A.scaler", &H_BCM4A_scaler);
    t_pi0sub->Branch("H.BCM4A.scalerCharge", &H_BCM4A_scalerCharge);
    t_pi0sub->Branch("H.BCM4A.scalerCurrent", &H_BCM4A_scalerCurrent);
    t_pi0sub->Branch("H.BCM4A.scalerChargeCut", &H_BCM4A_scalerChargeCut);
    t_pi0sub->Branch("H.BCM4A_Hel.scalerCharge", &H_BCM4A_Hel_scalerCharge);
    t_pi0sub->Branch("H.BCM4A_Hel.scalerCurrent", &H_BCM4A_Hel_scalerCurrent);
    t_pi0sub->Branch("H.BCM4A_Hel.scaler", &H_BCM4A_Hel_scaler);
    // photons from pi0 contamination
    Int_t N0, N1, N2;
    Double_t weight;
    vector<double> phXc;
    vector<double> phYc;
    vector<double> phPx;
    vector<double> phPy;
    vector<double> phPz;
    vector<double> phE;
    vector<double> phMx2;
    t_pi0sub->Branch("pi0.cont.Xc", &phXc);
    t_pi0sub->Branch("pi0.cont.Yc", &phYc);
    t_pi0sub->Branch("pi0.cont.trk.px", &phPx);
    t_pi0sub->Branch("pi0.cont.trk.py", &phPy);
    t_pi0sub->Branch("pi0.cont.trk.pz", &phPz);
    t_pi0sub->Branch("pi0.cont.trk.ene", &phE);
    t_pi0sub->Branch("pi0.cont.Mx2", &phMx2);
    t_pi0sub->Branch("pi0.cont.N0", &N0);
    t_pi0sub->Branch("pi0.cont.N1", &N1);
    t_pi0sub->Branch("pi0.cont.N2", &N2);
    t_pi0sub->Branch("pi0.cont.weight", &weight);

    // Output histograms__________________________________________________________
    TH1F *h_reactX = new TH1F("h_reactX", "H.react.x;Reaction X [cm];Counts", 400, -0.4, 0.1);
    TH1F *h_reactY = new TH1F("h_reactY", "H.react.y;Reaction Y [cm];Counts", 400, -0.2, 0.2);
    TH1F *h_reactZ = new TH1F("h_reactZ", "H.react.z;Reaction Z [cm];Counts", 400, -20, 20);

    TH1F *h_th = new TH1F("h_th", "H.gtr.th", 400, -2, 2);
    TH2F *hh_dp_ph = new TH2F("hh_dp_ph", "H.gtr.dp vs. H.gtr.ph;", 100, -0.05, 0.05, 300, -15, 15);

    TH1F *h_etottracknorm = new TH1F("h_etottracknorm", "H.cal.etottracknorm", 1000, 0, 10);
    TH1F *h_npeSum = new TH1F("h_npeSum", "H.cer.npeSum", 1000, 0, 10);
    TH1F *h_beta = new TH1F("h_beta", "H.hod.beta;#beta;Counts", 1500, 0, 1.5);
    TH1F *h_NpsTime = new TH1F("h_NpsTime", "NPS timing with offset correction (ampl. > 10 mV);Pulse time [ns];Number of events", 20000, -100, 100);

    TH1F *h_nclust = new TH1F("h_nclust", "Number of clusters;Number of clusters;Counts", 21, -0.5, 20.5);

    // Event loop
    Int_t nevt = nevt_T;
    if(max_events > 0 && max_events < nevt) nevt = max_events;
    // Int_t nevt = 10000; // for testing
    Int_t idxTW[3]; // index that corresponds to the time window: idxTW[i]=0->(-11,-5), idxTW[i]=1->(-3,3), idxTW[i]=2->(5,11)
    Double_t tmean_arr[3] = {-8., 0., 8.}; // time window array
    for(Int_t ievt = 0; ievt < nevt; ievt++){ // Event loop
        t_T->GetEntry(ievt);
        t_wf->GetEntry(indexArray[ievt]); // indexArray[i] is the original-entry number for the i-th smallest evt
        if(ievt%100000==0) cout << "Looking at entry = " << ievt << "  (" << 100.*ievt/nevt_T << "%)" << endl;
        
        if(evtNb_wf != evtNb_T){
            cerr<< "Error: global event number in T tree ("<<evtNb_T<<") and WF tree ("<<evtNb_wf<<") do not match!"<<endl;
            return;
        }

        bool flipTW = ievt % 2; // to invert the time-window sequence
        if(!flipTW){ idxTW[0] = 0; idxTW[1] = 1; idxTW[2] = 2; } 
        else{ idxTW[0] = 2; idxTW[1] = 1; idxTW[2] = 0; } // invert time window order when ievt is odd

        // Initialize for production tree
        nclust = 0; // for all pulse time window
        nclust_acc1 = 0; // pulse time in (-11, -5)
        nclust_coin = 0; // pulse time in (-3, 3)
        nclust_acc2 = 0; // pulse time in (5, 11)
        clusE.clear();
        clusSize.clear();
        clusX.clear();
        clusX_corr.clear();
        clusY.clear();
        clusY_corr.clear();
        clusT.clear();
        clusDepth.clear();
        trkPx.clear();
        trkPy.clear();
        trkPz.clear();
        trkE.clear();
        M.clear();
        Mx2.clear();
        // Initialize for pi0 contamination tree
        N0 = 0;
        N1 = 0;
        N2 = 0;
        weight = 1.;
        phXc.clear();
        phYc.clear();
        phPx.clear();
        phPy.clear();
        phPz.clear();
        phE.clear();
        phMx2.clear();

        if(abs(H_react_x) > 100 || abs(H_react_y) > 100 || abs(H_react_z) > 100){ 
            // Skip photon reconstruction for strange vertex postion of 1e+38
            // Fill the Tree then jump to next event without clustering
            t_prod->Fill();
            continue;
        }

        // Fill the histograms
        h_reactX->Fill(H_react_x);
        h_reactY->Fill(H_react_y);
        h_reactZ->Fill(H_react_z);
        h_th->Fill(H_gtr_th);
        hh_dp_ph->Fill(H_gtr_ph, H_gtr_dp);
        h_etottracknorm->Fill(H_cal_etottracknorm);
        h_npeSum->Fill(H_cer_npeSum);
        h_beta->Fill(H_hod_beta);
        
        // scattered electron four momenta
        Double_t H_gtr_e = sqrt(H_gtr_px*H_gtr_px+H_gtr_py*H_gtr_py+H_gtr_pz*H_gtr_pz+0.000511*0.000511);
        TLorentzVector kpvec(H_gtr_px, H_gtr_py, H_gtr_pz, H_gtr_e);

        for(int it = 0; it < 3; it++){ // Loop over three different time windows -> clustering for 3 times
            Double_t tmean = tmean_arr[idxTW[it]]; // central value of the time window, use idxTW[i] to select time window

            // Add pulse energy and timing for clustering (using the waveform tree)__________________________
            Int_t flatIdx = 0; // accumulated number of pulses in previous blocks
            for (size_t iblk=0; iblk<wfnpulse->size(); iblk++){
                Int_t nPulse = (*wfnpulse)[iblk];
                Int_t iblk_sim = bnConv_NewToOld(iblk); // block number with simulation numbering scheme
                TCaloBlock *block = caloev->AddBlock(iblk_sim); // add block with simulation numbering scheme
                if (nPulse > 0 && (*chi2)[iblk] > 0 && !caloMaskBlock[iblk]){ // if we want to read the pulse in this block
                    Int_t pulseIdx = flatIdx + 0; // index of the first pulse in this block
                    for (Int_t p = 0; p < nPulse; p++){ // Find the pulse closest to the mean time
                        if(abs((*wftime)[flatIdx + p]+timingOffset[iblk]-tmean) <= abs((*wftime)[pulseIdx]+timingOffset[iblk]-tmean)) pulseIdx = flatIdx + p; 
                    }
                    if(idxTW[it] == 1 && (*wfampl)[pulseIdx] > 10) h_NpsTime->Fill((*wftime)[pulseIdx]+timingOffset[iblk]); // Fill the histogram with timing offset correction

                    if(abs((*wftime)[pulseIdx]+timingOffset[iblk]-tmean) < 3) block->AddPulse((*wfampl)[pulseIdx] * coefPi0[iblk], (*wftime)[pulseIdx]+timingOffset[iblk]); // Add energy and timing to the block
                } // block selection
                flatIdx += nPulse;
            } // loop over 1080 blocks

            caloev->TriggerSim(clusTrsH); // Energy threshold of every 2x2 blocks
            caloev->DoClustering(tmean-3, tmean+3); // Time window cut (tmean-3,tmean+3) [ns] for clustering
            ev->SetCaloEvent(caloev);
            ev->SetVertex(H_react_x, H_react_y, H_react_z);

            if(idxTW[it] == 0) nclust_acc1 = caloev->GetNbClusters(); // Accidental#1 time window = -8+-3ns
            if(idxTW[it] == 1) nclust_coin = caloev->GetNbClusters(); // Production time window = 0+-3ns
            if(idxTW[it] == 2) nclust_acc2 = caloev->GetNbClusters(); // Accidental#2 time window = 8+-3ns

            for(int iclus = 0; iclus < caloev->GetNbClusters(); iclus++){
                caloev->GetCluster(iclus)->Analyze(1);
                TCaloCluster *cluster = caloev->GetCluster(iclus);
                const int prod_cluster_slot = clusE.size();

                clusE.push_back(cluster->GetEnergy());
                clusSize.push_back(cluster->GetClusSize());
                clusX.push_back(cluster->GetX());
                clusY.push_back(cluster->GetY());

                Float_t a_out, x_corr_out, y_corr_out;
                TLorentzVector photon = ev->GetPhoton(iclus, 7, 0, a_out, x_corr_out, y_corr_out);

                clusDepth.push_back(a_out);
                clusX_corr.push_back(x_corr_out);
                clusY_corr.push_back(y_corr_out);

                trkPx.push_back(photon.Px());
                trkPy.push_back(photon.Py());
                trkPz.push_back(photon.Pz());
                trkE.push_back(photon.E());

                // calculate cluster time
                Double_t sum_et = 0.;
                for(int iblk = 0; iblk < cluster->GetClusSize(); iblk++){
                    Double_t blockE = cluster->GetBlock(iblk)->GetEnergy(0);
                    Double_t blockT = cluster->GetBlock(iblk)->GetTime(0);
                    if(blockE > 0) sum_et+= blockE*blockT;
                    else(cout<<"Warning: block energy <= 0 ("<<blockE<<" GeV)"<<endl);
                }
                const Double_t cluster_t = sum_et/cluster->GetEnergy();
                clusT.push_back(cluster_t);

                Int_t seed_block_old = -1;
                Int_t seed_block_new = -1;
                Int_t seed_row = -1;
                Int_t seed_col = -1;
                Double_t seed_energy = -1.;
                for(int iblk = 0; iblk < cluster->GetClusSize(); iblk++){
                    TCaloBlock *block = cluster->GetBlock(iblk);
                    if(!block) continue;
                    const Double_t blockE = block->GetBlockEnergy();
                    if(blockE > seed_energy){
                        seed_energy = blockE;
                        seed_block_old = block->GetBlockNumber();
                    }
                }
                if(seed_block_old >= 0){
                    seed_block_new = bnConv_OldToNew(seed_block_old);
                    seed_row = seed_block_new / 30;
                    seed_col = seed_block_new % 30;
                }
                for(int iblk = 0; iblk < cluster->GetClusSize(); iblk++){
                    TCaloBlock *block = cluster->GetBlock(iblk);
                    if(!block) continue;
                    const Double_t memberE = block->GetBlockEnergy();
                    if(!(memberE > 0.)) continue;
                    const Int_t block_old = block->GetBlockNumber();
                    const Int_t block_new = bnConv_OldToNew(block_old);
                    sidecar << ievt << '\t'
                            << indexArray[ievt] << '\t'
                            << run_number << '\t'
                            << iseg << '\t'
                            << static_cast<Long64_t>(evtNb_T) << '\t'
                            << idxTW[it] << '\t'
                            << tmean << '\t'
                            << prod_cluster_slot << '\t'
                            << iclus << '\t'
                            << cluster->GetEnergy() << '\t'
                            << cluster->GetClusSize() << '\t'
                            << cluster->GetX() << '\t'
                            << cluster->GetY() << '\t'
                            << cluster_t << '\t'
                            << seed_block_new << '\t'
                            << seed_block_old << '\t'
                            << seed_row << '\t'
                            << seed_col << '\t'
                            << seed_energy << '\t'
                            << iblk << '\t'
                            << block_new << '\t'
                            << block_old << '\t'
                            << (block_new / 30) << '\t'
                            << (block_new % 30) << '\t'
                            << memberE << '\t'
                            << block->GetTime(0) << '\t'
                            << (memberE / cluster->GetEnergy()) << '\n';
                }
            } // loop of clusters

            if(caloev->GetNbClusters() == 1){
                TLorentzVector photon1 = ev->GetPhoton(0);
                Mx2.push_back((beam + p0 - kpvec - photon1).Mag2());
            } // clus==1

            else if(caloev->GetNbClusters() == 2){
                TLorentzVector photon1 = ev->GetPhoton(0);
                TLorentzVector photon2 = ev->GetPhoton(1);
                TLorentzVector pion_tlv = photon1 + photon2;

                M.push_back(pion_tlv.M());
                Mx2.push_back((beam + p0 - kpvec - pion_tlv).Mag2());

                // pi0 subtraction__________________________________________________________________
                if(enable_pi0sub && idxTW[it] == 1){ // coin timing
                    // acceptance cut for photons
                    Int_t nCol_inMgnt = 0; // number of columns under the shadow of magnet
                    if(290 < NPS_dist && NPS_dist < 310) nCol_inMgnt = 5; // col0-4 when NPS@300cm
                    else if(340 < NPS_dist && NPS_dist < 360) nCol_inMgnt = 4; // col0-3 when NPS@350cm
                    else if(390 < NPS_dist && NPS_dist < 410) nCol_inMgnt = 2; // col0-1 when NPS@400cm

                    Double_t x1 = caloev->GetCluster(0)->GetX();
                    Double_t y1 = caloev->GetCluster(0)->GetY();
                    Double_t ene1 = caloev->GetCluster(0)->GetEnergy();
                    
                    Double_t x2 = caloev->GetCluster(1)->GetX();
                    Double_t y2 = caloev->GetCluster(1)->GetY();
                    Double_t ene2 = caloev->GetCluster(1)->GetEnergy();

                    Double_t Xcut_min = -2.15*(15-1-nCol_inMgnt);
                    Double_t Xcut_max = 2.15*(15-1);
                    Double_t Ycut_min = -2.15*(18-1);
                    Double_t Ycut_max = 2.15*(18-1);

                    if(abs(pion_tlv.M()-0.1349766) < 3*pi0_sigm // pi0 mass window cut
                    && ene1 > 0.5 && ene2 > 0.5
                    && Xcut_min < x1 && x1 < Xcut_max && Ycut_min < y1 && y1 < Ycut_max
                    && Xcut_min < x2 && x2 < Xcut_max && Ycut_min < y2 && y2 < Ycut_max){
                        
                        for(int idc = 0; idc < ndecay; idc++){
                            Double_t cosalpha = 2.*gRandom->Rndm()-1.; // uniform polor angle cos(alpha) between -1 and 1
                            Double_t phi2 = 2.*TMath::Pi()*gRandom->Rndm(); // uniform azimuthal angle between 0 and 2*pi
                            // Decay in pion rest frame
                            Double_t e_photon = 0.5*pion_tlv.M();
                            Double_t px_photon = e_photon*TMath::Sin(TMath::ACos(cosalpha))*TMath::Cos(phi2);
                            Double_t py_photon = e_photon*TMath::Sin(TMath::ACos(cosalpha))*TMath::Sin(phi2);
                            Double_t pz_photon = e_photon*cosalpha;
                            TLorentzVector photonA(px_photon, py_photon, pz_photon, e_photon);
                            TLorentzVector photonB(-1*px_photon, -1*py_photon, -1*pz_photon, e_photon);
                            // Boost to lab frame
                            photonA.Boost(pion_tlv.BoostVector());
                            photonB.Boost(pion_tlv.BoostVector());
                            Double_t eneA = photonA.E();
                            Double_t eneB = photonB.E();
                            // Impact position on NPS surface
                            Double_t beta = TMath::ATan(photonA.Px() / photonA.Pz());
                            Double_t xcA = NPS_dist*TMath::Tan(beta-NPS_angle)-H_react_z*TMath::Sin(beta)/TMath::Cos(beta-NPS_angle);
                            Double_t ycA = (TMath::Sqrt(NPS_dist*NPS_dist+xcA*xcA)-H_react_z*(TMath::Cos(beta)+TMath::Sin(beta)*TMath::Tan(beta-NPS_angle))) * photonA.Py();
                            ycA = ycA/TMath::Sqrt(photonA.Px()*photonA.Px()+photonA.Pz()*photonA.Pz());

                            beta = TMath::ATan(photonB.Px() / photonB.Pz());
                            Double_t xcB = NPS_dist*TMath::Tan(beta-NPS_angle)-H_react_z*TMath::Sin(beta)/TMath::Cos(beta-NPS_angle);
                            Double_t ycB = (TMath::Sqrt(NPS_dist*NPS_dist+xcB*xcB)-H_react_z*(TMath::Cos(beta)+TMath::Sin(beta)*TMath::Tan(beta-NPS_angle)))*photonB.Py();
                            ycB = ycB/TMath::Sqrt(photonB.Px()*photonB.Px()+photonB.Pz()*photonB.Pz());

                            // if the decayed photon pass the acceptance cut
                            Bool_t passA = false;
                            Bool_t passB = false;

                            if(eneA > 0.5 
                            && Xcut_min < xcA && xcA < Xcut_max 
                            && Ycut_min < ycA && ycA < Ycut_max) passA = true;

                            if(eneB > 0.5 
                            && Xcut_min < xcB && xcB < Xcut_max 
                            && Ycut_min < ycB && ycB < Ycut_max) passB = true;

                            if(!passA && !passB) N0++; // no photon pass the acceptance cut
                            if(passA && passB) N2++; // both photons pass the acceptance cut
                            if((!passA && passB) || (passA && !passB)){ // only one photon pass the acceptance cut
                                N1++;
                                if(passA){
                                    Double_t Mx2_A = (beam + p0 - kpvec - photonA).Mag2();
                                    phXc.push_back(xcA);
                                    phYc.push_back(ycA);
                                    phPx.push_back(photonA.Px());
                                    phPy.push_back(photonA.Py());
                                    phPz.push_back(photonA.Pz());
                                    phE.push_back(eneA);
                                    phMx2.push_back(Mx2_A);
                                }
                                if(passB){
                                    Double_t Mx2_B = (beam + p0 - kpvec - photonB).Mag2();
                                    phXc.push_back(xcB);
                                    phYc.push_back(ycB);
                                    phPx.push_back(photonB.Px());
                                    phPy.push_back(photonB.Py());
                                    phPz.push_back(photonB.Pz());
                                    phE.push_back(eneB);
                                    phMx2.push_back(Mx2_B);
                                }
                            } // if only one photon pass
                        } // loop over ndecay
                        weight = 1./N2;
                        t_pi0sub->Fill();
                    } // pi0 mass window cut
                } // if it==1
            } // clus==2

            // reinitialization for next time window
            caloev->Reset();
        } // Loop over different time windows with mean time{-8,0,8}ns
        nclust = nclust_acc1+nclust_coin+nclust_acc2;
        h_nclust->Fill(nclust);
        t_prod->Fill();

    } // Event loop: production and accidental tree

    cout<<endl;
    cout<<"Number of event summary ===================="<<endl;
    cout<<"T tree: "<<nevt_T<<endl;
    cout<<"waveform tree: "<<nevt_wf<<endl;
    cout<<"production tree: "<<t_prod->GetEntries()<<endl;
    cout<<"pi0 contamination tree: "<<t_pi0sub->GetEntries()<<endl;
    cout<<"============================================"<<endl;
    cout<<endl;
    if(nevt_T != t_prod->GetEntries()){
        std::cerr<<"Error: number of events in T tree and production tree do not match!"<<endl;
    }

    // Save the output file__________________________________________________________
    outfile->cd();
    t_prod->Write();
    t_pi0sub->Write();
    
    h_reactX->Write();
    h_reactY->Write();
    h_reactZ->Write();
    h_beta->Write();
    h_etottracknorm->Write();
    h_npeSum->Write();
    h_th->Write();
    hh_dp_ph->Write();
    h_nclust->Write();
    h_NpsTime->Write();
    
    outfile->Close();
    sidecar.close();
}

// Functions to convert the numbering scheme of NPS
Int_t bnConv_OldToNew(int ibn_old) // Convert the PMT number to the current version
{
  const Int_t ncol = 30; // number of columns
  const Int_t nrow = 36; // number of rows
  Int_t irow = ibn_old % nrow;
  Int_t icol = 29 - ((ibn_old - irow) / nrow);
  Int_t ibn_new = ncol * irow + icol;

  return ibn_new;
}

Int_t bnConv_NewToOld(int ibn_new) // Convert the PMT number to the simulation version
{
  const Int_t ncol = 30; // number of columns
  const Int_t nrow = 36; // number of rows
  Int_t icol = 29 - (ibn_new % ncol);
  Int_t irow = (ibn_new - ibn_new % ncol) / ncol;
  Int_t ibn_old = nrow * icol + irow;

  return ibn_old;
}

bool readScalarFile(const TString &path, Double_t &val)
{
    std::ifstream in(path.Data());
    if (!(in >> val))
    {
        std::cerr << "Error reading scalar DB cache file " << path << "\n";
        return false;
    }
    return true;
}

bool readVectorFile(const TString &path, std::vector<Double_t> &vals, int expected)
{
    std::ifstream in(path.Data());
    if (!in)
    {
        std::cerr << "Error opening vector DB cache file " << path << "\n";
        return false;
    }
    vals.clear();
    Double_t val = 0.;
    while (in >> val)
        vals.push_back(val);
    if ((int)vals.size() != expected)
    {
        std::cerr << "Error: " << path << " has " << vals.size() << " entries; expected " << expected << "\n";
        return false;
    }
    return true;
}

bool readIntVectorFile(const TString &path, std::vector<Int_t> &vals, int expected)
{
    std::ifstream in(path.Data());
    if (!in)
    {
        std::cerr << "Error opening integer vector DB cache file " << path << "\n";
        return false;
    }
    vals.clear();
    Int_t val = 0;
    while (in >> val)
        vals.push_back(val);
    if ((int)vals.size() != expected)
    {
        std::cerr << "Error: " << path << " has " << vals.size() << " entries; expected " << expected << "\n";
        return false;
    }
    return true;
}
