/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Development version of the Structure Factor collective variable.
  This is a work in progress, be careful: edges are rough.

  Authors: Michele Invernizzi - https://github.com/invemichele
           Luigi Bonati       - https://github.com/luigibonati

  Please read and cite: 
  - "Collective variables for the study of crystallisation"
    Karmakar, Invernizzi, Rizzi, Parrinello - Mol. Phys. (2021)
  Additional reading:
  - "Deep learning the slow modes for rare events sampling"
    Bonati, Piccini, Parrinello - PNAS (2021)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Communicator.h"

#include <cmath>
#include <algorithm> //std::stable_sort, std::sort
#include <sstream>   //std::ostringstream

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR STRUCTURE_FACTOR_DESCRIPTOR_TEST
/*
Use as CVs the instantaneous per shell structure factor:
 S_k = 1/(N*m_k)\sum_{|\vec{k}|=k}|\rho_{\vec{k}}|^2

There is also the possibility of considering only certain atoms and a ficticious box edge L.
These options are mainly just vestigia from the past.

Optionally, one can compute only the peaks related to a given crystal structure based on miller indexes.
This is implemented for BCC,FCC and DIAMOND structures. It requires to specify both the STRUCTURE keyword and the number of repetitions of the unit cell (UNIT_CELLS). See second example below.

\par Examplexs

label: STRUCTURE_FACTOR_DESCRIPTOR_TEST NO_VIRIAL N2_MAX=30

label: STRUCTURE_FACTOR_DESCRIPTOR_TEST ACTIVE_SHELLS=27,72,99 STRUCTURE=DIAMOND UNIT_CELLS=3

*/
//+ENDPLUMEDOC

class StructureFactor_descriptor_test : public Colvar {

private:
  bool first_run_;
  bool no_virial_;
  unsigned NumParallel_; //number of parallel tasks
  unsigned rank_;

  unsigned NumAtom_;
  unsigned n2_max_;
  unsigned n_max_;
  std::vector<Vector> fwv_;
  std::vector<unsigned> fwv_mult_;
  double k_const_;

  // select based on miller indices
  bool use_structure_;
  int UnitCells_;
  string Structure_;
  std::vector<Value*> valueSk;

  void init_fwv(std::vector<unsigned>&);

public:
  StructureFactor_descriptor_test(const ActionOptions&);
  virtual void calculate();
  static void registerKeywords(Keywords& );
};

PLUMED_REGISTER_ACTION(StructureFactor_descriptor_test,"STRUCTURE_FACTOR_DESCRIPTOR_TEST")

void StructureFactor_descriptor_test::registerKeywords(Keywords& keys)
{
  Colvar::registerKeywords(keys);
  keys.add("optional","N2_MAX","the maximum square module of the indexing integer vector of the Fourier wave vectors (k) considered");
  keys.add("optional","K_MAX","calculate the structure factor up to this k value");
  keys.add("optional","ACTIVE_SHELLS","manually set which n2-shells will be considered");

  keys.add("atoms","ATOMS","calculate Fourier components using only these atoms. Default is to use all atoms");
  keys.add("optional","BOX_EDGE","manually set the edge L of the cubic box to be considered");
  keys.add("optional","NAME_PRECISION","set the number of digits used for components name");

  keys.add("optional","UNIT_CELLS","set the number of unit cells");
  keys.add("optional","STRUCTURE","choose the target structure");

  keys.addFlag("NO_VIRIAL",false,"skip the virial calculations, useful to speedup when a correct pressure is not needed (e.g. in NVT simulations)");
  keys.addFlag("SERIAL",false,"perform the calculation in serial even if multiple tasks are available");

  keys.addOutputComponent("Sk","default","the instantaneous structure factor averaged over a k-shell"); //FIXME not true!
  ActionWithValue::useCustomisableComponents(keys); //needed to have an unknown number of components
}

StructureFactor_descriptor_test::StructureFactor_descriptor_test(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao)
{
//parse and initialize:
  first_run_=true;

//- flags
  no_virial_=false;
  parseFlag("NO_VIRIAL",no_virial_);
  if(no_virial_)
    log.printf("  -- NO_VIRIAL: the virial contribution will not be calculated\n");

  NumParallel_=comm.Get_size();
  rank_=comm.Get_rank();
  log.printf("  number of parallel tasks: %d\n",NumParallel_);
  bool serial=false;
  parseFlag("SERIAL",serial);
  if (serial)
  {
    log.printf("  -- SERIAL: running without loop parallelization\n");
    NumParallel_=1;
    rank_=0;
  }

//- structure info
  use_structure_=false;
  Structure_="";
  parse("STRUCTURE",Structure_);

  if(!Structure_.empty())
  {
    use_structure_=true;
    log.printf("  selecting only the peaks of structure = %s\n",Structure_);
    UnitCells_=0;
    parse("UNIT_CELLS",UnitCells_);
    plumed_massert(UnitCells_>0,"if STRUCTURE is used the number of UNIT_CELLS must be specified");
    log.printf("  with a number of repetition of the unit cells (per side) = %d\n",UnitCells_ );
  }
 
//- edge L
  k_const_=1; //not elegant, but works...
  double box_edge=-1;
  parse("BOX_EDGE",box_edge);
  if (box_edge!=-1) //setting BOX_EDGE=-1 is equivalent to not set it
  {
    plumed_massert(box_edge>0,"BOX_EDGE must be greater than zero");
    k_const_=2*PLMD::pi/box_edge;
    log.printf("  considered box edge is L = %g\n",box_edge);
  }

//- active shells
  n2_max_=0;
  parse("N2_MAX",n2_max_);
  double k_max=0;
  parse("K_MAX",k_max);
  if (k_max!=0)
  {
    plumed_massert(n2_max_==0,"either set the maximum through K_MAX or through N2_MAX");
    plumed_massert(k_const_!=1,"need to explicitly set BOX_EDGE in order to use K_MAX keyword");
    n2_max_=std::floor(pow(k_max/k_const_,2));
    log.printf("  setting N2_MAX=floor((BOX_EDGE/(2pi)*K_MAX)^2) with K_MAX = %g\n",k_max);
  }
  std::vector<unsigned> picked_shells; //only the picked shells
  std::vector<unsigned> active_shells; //all shells, 1 or 0 if active or not
  parseVector("ACTIVE_SHELLS",picked_shells);
  if (picked_shells.size()==0)
  {
    plumed_massert(n2_max_>0,"if ACTIVE_SHELLS is not used, N2_MAX must be set");
    active_shells.resize(n2_max_,1);
    log.printf("  all n2-shells are considered, up to N2_MAX = %d\n",n2_max_);
  }
  else
  {
    plumed_massert(n2_max_==0,"if ACTIVE_SHELLS is used, N2_MAX cannot be manually set");
    std::sort(picked_shells.begin(),picked_shells.end()); //just to be sure
    n2_max_=picked_shells.back(); //last element is n2_max_
    log.printf("  -- ACTIVE_SHELLS: not all shells will be considered\n    active n2-shells:");
    active_shells.resize(n2_max_,0);
    for (unsigned i=0; i<picked_shells.size(); i++)
    {
      plumed_massert(picked_shells[i]>0,"the ACTIVE_SHELLS must start from 1 not from 0");
      active_shells[picked_shells[i]-1]++;
      log.printf("  %d",picked_shells[i]);
    }
    log.printf("\n");
  }

//- initialize Fourier wave vectors fwv, and its multiplicity fwv_mult
  init_fwv(active_shells);// uses n2_max_ and NumParallel_
  log.printf("  total number of indipendent k frequencies: %d\n",fwv_.size());

//add components
//  valueSk.resize(n2_max_);
  valueSk.resize(fwv_.size());
  std::ostringstream oss;
  unsigned name_precision=7;
  parse("NAME_PRECISION",name_precision);
  oss.precision(name_precision);
  log.printf("  components name are k value, with NAME_PRECISION = %d\n",name_precision);
  if (k_const_==1)
    log.printf("  --WARNING: since BOX_EDGE is not specified, the components name are equal to sqrt(n2)\n");
//  for (unsigned n2=0; n2<n2_max_; n2++)
  for (unsigned k=0; k<fwv_.size(); k++)
  {
//    if(active_shells[n2])
//    {
    oss.str("");
//      oss<<"Sk-"<<k_const_*sqrt(n2+1);
    oss<<"Sk-"<<"["<<fwv_[k][0]<<"_"<<fwv_[k][1]<<"_"<<fwv_[k][2]<<"]";
    addComponentWithDerivatives(oss.str());
    componentIsNotPeriodic(oss.str());
//      valueSk[n2]=getPntrToComponent(oss.str());
    valueSk[k]=getPntrToComponent(oss.str());
//    }
  }

//finish the parsing: get the atoms...
  vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  NumAtom_=atoms.size();
  if (NumAtom_==0) //default is to use all the atoms
  {
    NumAtom_=plumed.getAtoms().getNatoms();
    atoms.resize(NumAtom_);
    for(unsigned j=0; j<NumAtom_; j++)
      atoms[j].setIndex(j);
  }
  requestAtoms(atoms);//this must stay after the addComponentWithDerivatives otherwise segmentation violation
  log.printf("  over a total of N_tot=%d, considering a number of atoms N=%d\n",plumed.getAtoms().getNatoms(),NumAtom_);
  log.printf("  -- TEST version. single components square root ---\n");

//parsing finished
  checkRead();
}

// calculator
void StructureFactor_descriptor_test::calculate()
{
//this exists only because the function getBox() does not work before here
  if (first_run_)
  {
    first_run_=false;
    //assuming an orthorombic cubic box. up to now there is no need for a broader compatibility
    const bool is_cubic=(abs(getBox()[0][0]-getBox()[1][1])<1e-3 && abs(getBox()[1][1]-getBox()[2][2])<1e-3);
//    plumed_massert(is_cubic,"the simulation box must be orthorombic and cubic");
    const double box_edge=pow(getBox().determinant(),1./3.);
    if (k_const_==1)
      k_const_=2*PLMD::pi/box_edge;
    //print log info
    log.printf("------------------------------------------------------------------------------------\n");
    log.printf("First run:\n");
    if (!is_cubic)
      log.printf(" --WARNING: the simulation box is NOT cubic!!\n");
    log.printf("  The simulation box (which should be cubic) has an edge L = %g\n",box_edge);
    log.printf("------------------------------------------------------------------------------------\n");
    log.printf("internal coordinates will be used, thus k_const_=2*pi\n");
    k_const_=2*PLMD::pi;
  }

//build arrays to store per axis phases, as complex numbers:
//  - each row stores a different power, starting from zero: x_axis[n][j]=exp(-i*2pi/L*n*X_j)
//  - even indexes are the real part and odd ones the imaginary
// trivial sin/cos calculations was roughly two times slower (derivatives taken away),
// while the use of std::complex was just slightly slower.
  const unsigned size=2*NumAtom_*(1+n_max_);
  std::vector<double> x_axis(size);
  std::vector<double> y_axis(size);
  std::vector<double> z_axis(size);
  for(unsigned j=0; j<NumAtom_; j++)
  {
    x_axis[2*j]=1;
    y_axis[2*j]=1;
    z_axis[2*j]=1;
  }
  for(unsigned j=0; j<NumAtom_; j++)
  {
    const unsigned index_1=2*(NumAtom_+j);
//    const double x_arg=-1*k_const_*getPosition(j)[0];
//    const double y_arg=-1*k_const_*getPosition(j)[1];
//    const double z_arg=-1*k_const_*getPosition(j)[2];
    Vector pos_j=getPbc().realToScaled(getPosition(j));
    const double x_arg=-1*k_const_*pos_j[0];
    const double y_arg=-1*k_const_*pos_j[1];
    const double z_arg=-1*k_const_*pos_j[2];
    x_axis[index_1]=cos(x_arg); x_axis[index_1+1]=sin(x_arg);
    y_axis[index_1]=cos(y_arg); y_axis[index_1+1]=sin(y_arg);
    z_axis[index_1]=cos(z_arg); z_axis[index_1+1]=sin(z_arg);
  }
  for (unsigned n=2; n<=n_max_; n++)
  {
    for(unsigned j=0; j<NumAtom_; j++) //FIXME a better cache handling would probably speed up...
    {
      const unsigned index_1=2*(NumAtom_+j);
      const unsigned index_n=2*(NumAtom_*n+j);
      const unsigned index_n1=2*(NumAtom_*(n-1)+j);
      x_axis[index_n] = x_axis[index_1]*x_axis[index_n1] - x_axis[index_1+1]*x_axis[index_n1+1];
      x_axis[index_n+1]=x_axis[index_1]*x_axis[index_n1+1]+x_axis[index_1+1]*x_axis[index_n1];
      y_axis[index_n] = y_axis[index_1]*y_axis[index_n1] - y_axis[index_1+1]*y_axis[index_n1+1];
      y_axis[index_n+1]=y_axis[index_1]*y_axis[index_n1+1]+y_axis[index_1+1]*y_axis[index_n1];
      z_axis[index_n] = z_axis[index_1]*z_axis[index_n1] - z_axis[index_1+1]*z_axis[index_n1+1];
      z_axis[index_n+1]=z_axis[index_1]*z_axis[index_n1+1]+z_axis[index_1+1]*z_axis[index_n1];
    }
  }

//now can build the Fourier components
//  std::vector<double> Sk(n2_max_,0);
  std::vector<double> Sk(fwv_.size(),0);
  std::vector<double> d_Sk;
  if (!doNotCalculateDerivatives())
    d_Sk.resize(3*NumAtom_*fwv_.size(),0);
//    d_Sk.resize(3*NumAtom_*n2_max_,0);
  for (unsigned int k=rank_; k<fwv_.size(); k+=NumParallel_) //if NumParallel_==1 is a normal serial loop
  {
//    const unsigned n2=fwv_[k].modulo2()-1;
    const unsigned index_nx=fwv_[k][0]*NumAtom_;
    const int sign_y=(fwv_[k][1]<0 ? -1 : 1);
    const unsigned index_ny=fwv_[k][1]*NumAtom_*sign_y;
    const int sign_z=(fwv_[k][2]<0 ? -1 : 1);
    const unsigned index_nz=fwv_[k][2]*NumAtom_*sign_z;
    double ReRho_k=0;
    double ImRho_k=0;
    std::vector<double> cos_kR(NumAtom_,0);
    std::vector<double> sin_kR(NumAtom_,0);
    for(unsigned j=0; j<NumAtom_; j++)
    {
      const double r_x=x_axis[2*(index_nx+j)];
      const double i_x=x_axis[2*(index_nx+j)+1];
      const double r_y=y_axis[2*(index_ny+j)];
      const double i_y=y_axis[2*(index_ny+j)+1]*sign_y;
      const double r_z=z_axis[2*(index_nz+j)];
      const double i_z=z_axis[2*(index_nz+j)+1]*sign_z;
      cos_kR[j]=r_x*r_y*r_z-r_x*i_y*i_z-i_x*r_y*i_z-i_x*i_y*r_z;
      sin_kR[j]=i_x*i_y*i_z-i_x*r_y*r_z-r_x*i_y*r_z-r_x*r_y*i_z;
      ReRho_k+=cos_kR[j];
      ImRho_k-=sin_kR[j];
    }
//    Sk[n2]+=pow(ReRho_k,2)+pow(ImRho_k,2);
    Sk[k]+=pow(ReRho_k,2)+pow(ImRho_k,2);
    if (!doNotCalculateDerivatives())
    {
      for(unsigned j=0; j<NumAtom_; j++)
      {
        const double d_Rho_kj=sin_kR[j]*ReRho_k+cos_kR[j]*ImRho_k;
//        d_Sk[3*(n2*NumAtom_+j)+0]+=d_Rho_kj*fwv_[k][0];
//        d_Sk[3*(n2*NumAtom_+j)+1]+=d_Rho_kj*fwv_[k][1];
//        d_Sk[3*(n2*NumAtom_+j)+2]+=d_Rho_kj*fwv_[k][2];
        d_Sk[3*(k*NumAtom_+j)+0]+=d_Rho_kj*fwv_[k][0];
        d_Sk[3*(k*NumAtom_+j)+1]+=d_Rho_kj*fwv_[k][1];
        d_Sk[3*(k*NumAtom_+j)+2]+=d_Rho_kj*fwv_[k][2];
      }
    }
  }
  if(NumParallel_>1)
  {
    comm.Sum(Sk);
    if (!doNotCalculateDerivatives())
      comm.Sum(d_Sk); //doesn't work with std::vector<Vector>
  }

//set the CV value and derivatives
//  for (unsigned n2=0; n2<n2_max_; n2++)
//  {
//    const double norm=NumAtom_*fwv_mult_[n2];
//    if (norm!=0) //some shells are empty or not active
//    {
//      valueSk[n2]->set(Sk[n2]/norm);
//      if (!doNotCalculateDerivatives())
//      {
//        for(unsigned i=0; i<3*NumAtom_; i++)
//          valueSk[n2]->setDerivative(i,(-2*k_const_/norm)*d_Sk[3*NumAtom_*n2+i]);
//        if (!no_virial_)
//          setBoxDerivativesNoPbc(valueSk[n2]);
//      }
//    }
//  }
  for (unsigned k=0; k<fwv_.size(); k++)
  {
//    valueSk[k]->set(Sk[k]/NumAtom_);
    const double norm_Sk=std::sqrt(Sk[k]/NumAtom_);
    valueSk[k]->set(norm_Sk);
    if (!doNotCalculateDerivatives())
    {
//      for(unsigned i=0; i<3*NumAtom_; i++)
//        valueSk[k]->setDerivative(i,(-2*k_const_/NumAtom_)*d_Sk[3*NumAtom_*k+i]);//FIXME is the factor 2 still correct?
      for(unsigned j=0; j<NumAtom_; j++)
      {
//        Vector d_Sk_j((-2*k_const_/NumAtom_)*d_Sk[3*(NumAtom_*k+j)],(-2*k_const_/NumAtom_)*d_Sk[3*(NumAtom_*k+j)+1],(-2*k_const_/NumAtom_)*d_Sk[3*(NumAtom_*k+j)+2]);
        const double pref_k=(-2*k_const_/NumAtom_)*0.5/norm_Sk;
        Vector d_Sk_j(pref_k*d_Sk[3*(NumAtom_*k+j)],pref_k*d_Sk[3*(NumAtom_*k+j)+1],pref_k*d_Sk[3*(NumAtom_*k+j)+2]);
        setAtomsDerivatives(valueSk[k],j,matmul(getPbc().getInvBox(),d_Sk_j));
        setAtomsDerivatives(valueSk[k],j,matmul(getPbc().getInvBox(),d_Sk_j));
      }
//      if (!no_virial_)
//        setBoxDerivativesNoPbc(valueSk[k]);
    }
  }
}

void StructureFactor_descriptor_test::init_fwv(std::vector<unsigned>& active_shells)
{
//initialize support variable n_maxs so that n_maxs[n2]^2<=n2_max_-n2
  std::vector<unsigned> n_maxs(n2_max_+1);
  for (unsigned n2=0; n2<=n2_max_; n2++)
    n_maxs[n2]=floor(sqrt(n2_max_-n2));
  n_max_=n_maxs[0];

//build the Fourier wave vectors indexes vector fwv_
//initialize fwv_mult_ so that it stores the multiplicity of k vector in a given shell active
  fwv_mult_.resize(n2_max_);
  int start;
  for (unsigned nx=0; nx<=n_max_; nx++)
  {
    start=(nx==0 ? 0 : -1*n_maxs[nx*nx]);
    for (int ny=start; ny<=(int)n_maxs[nx*nx]; ny++)
    {
      const unsigned nx2ny2=nx*nx+ny*ny;
      start=(nx2ny2==0 ? 1 : -1*n_maxs[nx2ny2]); //from 1 to avoid (0,0,0)
      for(int nz=start; nz<=(int)n_maxs[nx2ny2]; nz++)
      {
        const unsigned n2=nx2ny2+nz*nz-1;
        if (active_shells[n2])
        {
          bool select_k=false;
          if (use_structure_)
          {
	        //check if remainder with respect to # of unit cell  is zero
            if (nx%UnitCells_ == 0 && ny%UnitCells_ == 0 && nz%UnitCells_ ==0)
            {
              int h = nx/UnitCells_;
              int k = ny/UnitCells_;
              int l = nz/UnitCells_;
              
              //check if satisfies the condition for miller indexes	
              if (Structure_=="BCC"){
                if ( (h+k+l)%2 == 0 )
                  select_k=true;
              }else if (Structure_=="FCC"){
                if ( (h%2==0 && k%2==0 && l%2==0 ) || ((h+1)%2==0 && (k+1)%2==0 && (l+1)%2==0 ) )
                  select_k=true;
              }else if (Structure_=="DIAMOND"){
                if ( (h+k+l)%4 == 0 || (h+k+l - 1 )%2 == 0)
                  select_k=true;
                }else{
                  plumed_massert(Structure_=="","Unkown structure type"); //TODO	
                }
            }
          } 
          else 
          {
            select_k=true;
          }

          if (select_k){
            Vector new_point(nx,ny,nz);
            fwv_.push_back(new_point);
            fwv_mult_[n2]++; //inactive shells will have zero multiplicity
          }
        }
      }
    }
  }
//ordered is nicer //FIXME is it still useful to sort it?
  std::stable_sort( fwv_.begin(),fwv_.end(),
  [](const Vector& a,const Vector& b) {return a.modulo2()<b.modulo2();} );
//print some info in log
  log.printf(" Here are the Fourier wave vectors:");
  int n2=0;
  for (unsigned k=0; k<fwv_.size(); k++)
  {
    if(fwv_[k].modulo2()>n2)
    {
      n2=fwv_[k].modulo2();
      log.printf("  n2=%d :\n",n2);
    }
    log.printf("   % g,% g,% g\n",fwv_[k][0],fwv_[k][1],fwv_[k][2]);
  }
}

}
}

/***********************************************

  Here are the total number of Fourier wave
  vectors k given different N2_MAX:

      n2_max  fwv_.size()
         1        3
         2        9
         3       13
         4       16
         5       28
         6       40
         7       40
         8       46
         9       61
        10       73
      [...]    [...]

************************************************/
