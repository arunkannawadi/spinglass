/* Header files that contains all the functions necessary for entanglement computation of spin-glass */
/* This version uses integers as basis */


#include<iostream>
#include<ctime>
#include<fstream>
#include<iomanip>
#include<bitset>
#include<stdlib.h>
#include<math.h>
#include<sstream>
#include<string>
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>
#include<boost/multiprecision/cpp_int.hpp>

using namespace std; 
using namespace Eigen;
using namespace boost::multiprecision;

#define root2 1.414213562373095
#define PI 3.141592653589793238462

unsigned int ctz128(int128_t num)
{
 unsigned int ctz = 0;
 while(((num>>(ctz+1))<<(ctz+1)==num)&(ctz<128))
    ctz++; 
 return ctz; 
}

int128_t *definite_particle_basis(int m, int LCm)
{
 int i,k;
 int128_t v,w,t;

 //states = (unsigned long long int*)malloc(LCm*sizeof(unsigned long long int));
 int128_t *states = new int128_t[LCm]; 
 v = (1 << m) - 1;
 k = 0;
 
 while(k<LCm)
 {
  states[k++] = v;
  t = v | (v-1);
  v = (t+1) | (((~t & -~t) -1) >> (ctz128(v)+1));
 }
 
 return states;
 
}

inline void flipull(unsigned long long &n,unsigned long long K)
{  
  n = n&K ? n&(~K) : n|K ;
}

inline void flip(unsigned int &n, unsigned int K)
{
  n = n&K ? n&(~K) : n|K ;
}

inline void flip_uint128(int128_t &n, int128_t K)
{
  n = n&K ? n&(~K) : n|K ;
}

inline void reset(int128_t &n,int128_t K)
{
  n = n&(~K);
}

inline void set(int128_t &n,int128_t K)
{
  n = n|K;
}

/*
inline void setequal_ull(unsigned long long &n, unsigned long long K, unsigned int binary)
{
  n = binary==0 ? n&(~K) : n|K;
}

inline void setequal(unsigned int &n, unsigned int K, int128_t binary)
{
  n = binary==0 ? n&(~K) : n|K;
}

inline void setequal_uint128(unsigned int &n, unsigned int K, int128_t binary)
{
  n = binary==0 ? n&(~K) : n|K;
}
*/

inline unsigned int access_bit(int128_t n, int128_t K)
{
  return (n&K) ? 1 : 0;
}

double VNE1(VectorXd state, int128_t *basis, int L, int m, int LCm)
{
 double disc, lambda, VNE, avgVNE, sumVNE=0, x=0,y=0,z=0;
 int i,j,l;
 int128_t ll;
 
 for(l=0,ll=1;l<L;l++,ll=ll<<1)
  {
   x = 0; y = 0; z = 0;
   for(i=0;i<LCm;i++)
     {
      if(basis[i]&ll)
        x += state[i]*state[i];
      else
        y += state[i]*state[i];
        
      for(j=0;j<i;j++)
       {
         if(basis[i]&basis[j]&ll==0)
          z += state[i]*state[j];
       }    
     }
   
   disc = sqrt(4*z*z + (x-y)*(x-y));
   lambda = 0.5*(x+y+disc); VNE = -lambda*log(lambda);
   lambda = 0.5*(x+y-disc); VNE -= lambda*log(lambda);   
   
   sumVNE += VNE;  
  }
  
  avgVNE = sumVNE/L;
  return avgVNE; 
}

void average_concurrence(double *entanglement, VectorXd state, int128_t *basis, int L, int m, int LCm)
{
  int i,j,k,l;
  unsigned int pc=0; 
  int128_t ii,jj,kk,ll;
  int nchoosek(int,int);
  void flip(int*,int);
           
  double v,w,x,y,z;
  double disc, lambda, VNE, avgVNE, PC, C, avgC, avgTangle, LN2, avgLN2, sumLN2=0,sumTangle=0,sumC=0, sumVNE = 0; 
  for(i=1;i<L;i++)
   for(j=0;j<i;j++)
   {
    ii = 1; ii = ii<<i;
    jj = 1; jj = jj<<j;
    v=0; y=0; z=0; w=0; x=0;
    for(k=0;k<LCm;k++)
     {
          
      switch((((basis[k]&ii)==0)<<1) | ((basis[k]&jj)==0))
      {
       case 3: {v += state(k)*state(k); break;}
       case 0: {y += state(k)*state(k); break;}
       case 1: 
              { w += state(k)*state(k);
              flip_uint128(basis[k],ii); flip_uint128(basis[k],jj);
              #pragma omp parallel for
	      for(l=0;l<k;l++)
       	      {
	        if(basis[k]==basis[l])
        	 z += state(k)*state(l);
              }  
              flip_uint128(basis[k],ii); flip_uint128(basis[k],jj);
              break;
              }
      case 2:
              {
              x += state(k)*state(k);
              flip_uint128(basis[k],ii); flip_uint128(basis[k],jj);    
	      #pragma omp parallel for
	      for(l=0;l<k;l++)
       	      {
	        if(basis[k]==basis[l])
        	 z += state(k)*state(l);
              }  
              flip_uint128(basis[k],ii); flip_uint128(basis[k],jj);
              break;
              }
      default: {cout << "OOPS! Switch-case method has failed!"; }
      }
      /*
      if(((basis[k]&ii)==0)&((basis[k]&jj)==0))
       { v += state(k)*state(k); }
      else if(((basis[k]&ii)==0)&((basis[k]&jj)>0))
        x += state(k)*state(k);
      else if(((basis[k]&ii)>0)&((basis[k]&jj)==0))
        w += state(k)*state(k);
      else if(((basis[k]&ii)>0)&((basis[k]&jj)>0))    
        y += state(k)*state(k);

      flip_uint128(basis[k],ii); flip_uint128(basis[k],jj);    
      for(l=0;l<k;l++)
       {
        if(basis[k]==basis[l])
          z += state(k)*state(l);
        } 
       
       flip_uint128(basis[k],ii); flip_uint128(basis[k],jj);
     */      
     }

     PC = abs(z)-sqrt(v*y);
     if(PC>0)
     {
      pc++;
      C = 2*PC;
      disc = sqrt(4*z*z+(v-y)*(v-y));
      LN2 = log(w+x+disc);
     }
     else
     {
      C = 0;
      LN2 = 0;
     }
//     cout << " \t C = " << C ;
     sumC += C;
     sumTangle += C*C;
     
     // LN2 = log(w+x+0.5*(v+y+sqrt(4*z*z+(v-y)*(v-y)))+0.5*abs(v+y-sqrt(4*z*z+(v-y)*(v-y))));
     sumLN2 += LN2;
     
     // VNE  
     disc = sqrt(4*z*z + (w-x)*(w-x));
     VNE = (v>0) ? -v*log(v) : 0;
     VNE += (y>0) ? -y*log(y) : 0;

     lambda = 0.5*(w+x+disc); VNE += (lambda>0) ? -lambda*log(lambda) : 0;
     lambda = 0.5*(w+x-disc); VNE += (lambda>0) ? -lambda*log(lambda) : 0;
     
     sumVNE += VNE;
   }
   
    int LC2 = nchoosek(L,2);
    
    avgC = sumC/LC2;
    avgTangle = sumTangle/LC2;
    avgLN2 = sumLN2/LC2;
    avgVNE = sumVNE/LC2; 
    
    
    // double *entanglement = new double[3];
    entanglement[0] = avgC;
    entanglement[1] = avgTangle;
    entanglement[2] = avgLN2;
    entanglement[3] = ((double)pc)/LC2;
    entanglement[4] = avgVNE;
    
    // return entanglement;
}

/*
void LN3(double *q3_entanglement, VectorXd state, int128_t* basis, int L, int m, int LCm)
{
 SelfAdjointEigenSolver<MatrixXd> es(8);
 bitset<3> B,B1, B2;
 unsigned int b,b1, b2, n1,n2,pp,p,i,j,k,l,l1,l2;
 int128_t ii,jj,kk;
 int nchoosek(int,int);
 int LC3 = nchoosek(L,3);
 
 double Lambda, VNE, LN3, avgVNE, avgLN3, sumVNE = 0, sumLN3=0;
 MatrixXd rho = MatrixXd::Zero(8,8);
 VectorXd eigenvals;
 
 // i,j,k - Qubit indices 
 // l1,l2  - Basis indices 
 
 for(i=2,ii=4;i<L;i++,ii=ii<<1)
  for(j=1,jj=2;j<i;j++,jj=jj<<1)
   for(k=0, kk=1;k<j;k++,kk=kk<<1)
   {
    for(p=0,pp=1;p<=3;p++,pp=pp<<1)
    {
     for(n1=0;n1<8;n1++)
      for(n2=0;n2<8;n2++) 
       rho(n1,n2) = 0;
     
     for(l1=0;l1<LCm;l1++)
     {
      b1 = 0;
      setequal(b1,1<<0,basis[l1]&kk);
      setequal(b1,1<<1,basis[l1]&jj);
      setequal(b1,1<<2,basis[l1]&ii);
     
      // B1.set(1,basis[l1]);
      // B1.set(2,basis[l1]&ii);
     
      reset(basis[l1],ii); reset(basis[l1],jj); reset(basis[l1],kk);
      
      // b1 = (int)B1.to_ulong();
      rho(b1,b1) += state(l1)*state(l1);
          
      for(l2=0;l2<l1;l2++)
      {
       //   B2.set(0,basis[l2][k]);        B2.set(1,basis[l2][j]);       B2.set(2,basis[l2][i]);
       
       b2 = 0;
       setequal(b2,1<<0,basis[l2]&kk);
       setequal(b2,1<<1,basis[l2]&jj);
       setequal(b2,1<<2,basis[l2]&ii);
       
       reset(basis[l2],ii); reset(basis[l2],jj); reset(basis[l2],kk);
      
       if(basis[l2]==basis[l1])
       {
        if(!((b2&pp)==(b1&pp)))
        { flip(b2,pp); flip(b1,pp); }
         
        // b1 = (int)B1.to_ulong();
        // b2 = (int)B2.to_ulong();
        rho(b1,b2) += state(l1)*state(l2);
        rho(b2,b1) = rho(b1,b2);
        
        if(!((b2&pp)==(b1&pp)))
         { flip(b2,pp); flip(b1,pp); }
       }
       
       setequal_uint128(basis[l2],ii,b2&(1<<2)); setequal_uint128(basis[l2],jj,b2&(1<<1)); setequal_uint128(basis[l2],kk,b2&(1<<0)); 
       b2 = 0;
      }
      setequal_uint128(basis[l1],ii,b1&(1<<2)); setequal_uint128(basis[l1],jj,b1&(1<<1)); setequal_uint128(basis[l2],kk,b1&(1<<0));              
      b1 = 0;
     }
                   
     // Diagonalize PPT here //
     es.compute(rho,EigenvaluesOnly);
     eigenvals = es.eigenvalues();

    if(p<3)
    {
     for(Lambda=0,n1=0;n1<8;n1++)
       { Lambda += abs(eigenvals[n1]); }
       
     LN3 = log(Lambda);
     sumLN3 += LN3;
    }
    else
    {
      for(VNE=0,n1=0;n1<8;n1++)
       if(eigenvals[n1]>0)
        VNE += -eigenvals[n1]*log(eigenvals[n1]);
      
      sumVNE += VNE;  
    }
    }     
   }
   avgLN3 = sumLN3/(3*LC3);
   avgVNE = sumVNE/(LC3); 
   
   q3_entanglement[0] = avgLN3;
   q3_entanglement[1] = avgVNE;
}
*/

double PR(VectorXd state)
{
 double ipr, pr;
 int k,n;
 
 n = state.size();
 for(ipr=0,k=0;k<n;k++)
  ipr += pow(state(k),4);
 pr = 1.0/ipr;
 return pr;
}

MatrixXd hamiltonian(double** J, int128_t *basis, int m, int L,int LCm)
{
 
  const unsigned int constL = L;  
  int i,j,k,n1,n2,n0;
  int128_t N1, N2, N0;
  int128_t u,v,w,x,y;
  int128_t one;
  // one.set(0);
    
  double tmp, sigmaJ;
  sigmaJ = 0;
  for(i=1;i<L;i++)
     for(j=0;j<i;j++)
       sigmaJ = sigmaJ + J[i][j];
           
  MatrixXd H = MatrixXd::Zero(LCm,LCm);
  
   for(i=0;i<LCm;i++)
     {
      for(j=LCm-1;j>i;j--)
      {
       u = basis[i];
       v = basis[j];       

       for(n1=0,N1=1;access_bit(u,N1)==access_bit(v,N1);n1++,N1=N1<<1);
       for(n2=n1+1, N2=(N1<<1);access_bit(u,N2)==access_bit(v,N2);n2++,N2=N2<<1);
       
       for(n0=n2+1,N0=1<<(n2+1);n0<L;n0++,N0=N0<<1)
         if(!(access_bit(u,N0)==access_bit(v,N0)))
          n0=L+2;
       if(n0==L)
       {       
         H(i,j) = 2*J[n1][n2];
         H(j,i) = 2*J[n1][n2];
       }  
      }
      
      H(i,i) = 0;
      
      for(tmp=0,k=0;k<LCm;k++)
         tmp -= H(i,k) ;
      H(i,i) = tmp + sigmaJ;         
     }
     
     return H;
}

/* For generating J */
double **symm_gauss_rand_mat_gen(int dim, float sigma=0.0)
{
  int i,j;
  double** randmat;
  double r,u,v,s,x,y,z;

  randmat = (double**)malloc(dim*sizeof(double*));
  for(i=0;i<dim;i++)
    randmat[i] = (double*)malloc(dim*sizeof(double));
 if(sigma>=0)
 {
  for(i=0;i<dim;i++)
    {
      for(j=0;j<i;j++)
	{
	  do
	    {
	      u = 2*((double)rand())/RAND_MAX -1;
	      v = 2*((double)rand())/RAND_MAX -1;
	      s = u*u + v*v ;
	      x = u*sqrt(-2*log(s)/s);
	      y = v*sqrt(-2*log(s)/s);
	      z = (x+y)/root2;   
	    } while(s>=1);      
	  
	  r = (dim/PI)*sin((i-j)*PI/dim);
	  
	  randmat[i][j] = z/pow(r,sigma);
	  randmat[j][i] = randmat[i][j];
	}
      randmat[i][i] = 0;
    }
  }
  else if(sigma=-1.0)
  {
    for(i=0;i<dim;i++)
     for(j=0;j<dim;j++)
       randmat[i][j] = 0.0;
       
    for(i=0;i<dim-1;i++)
    {
     do
     {
      u = 2*((double)rand())/RAND_MAX -1;
      v = 2*((double)rand())/RAND_MAX -1;
      s = u*u + v*v ;
      x = u*sqrt(-2*log(s)/s);
      y = v*sqrt(-2*log(s)/s);
      z = (x+y)/root2;   
     }while(s>=1);
     
     randmat[i][i+1] = z;
     randmat[i+1][i] = z;
    }
   
     do
     {
      u = 2*((double)rand())/RAND_MAX -1;
      v = 2*((double)rand())/RAND_MAX -1;
      s = u*u + v*v ;
      x = u*sqrt(-2*log(s)/s);
      y = v*sqrt(-2*log(s)/s);
      z = (x+y)/root2;   
     }while(s>=1);
     
     randmat[i][0] = z;
     randmat[0][i] = z;
   
  }
  
  return randmat;
}

VectorXd gauss_rand_vec_gen(int len)
{
  int i,j;
  VectorXd vector = VectorXd::Zero(len);
  double norm,u,v,s,x,y,z;

  for(norm=0,i=0;i<len;i++)
    {
     do
	{
          u = 2*((double)rand())/RAND_MAX -1;
	  v = 2*((double)rand())/RAND_MAX -1;
	  s = u*u + v*v ;
	  x = u*sqrt(-2*log(s)/s);
	  y = v*sqrt(-2*log(s)/s);
	  z = (x+y)/root2;   
	} while(s>=1);      
	  
     vector(i) = z;	
     norm += z*z;
   }  
	
  for(i=0;i<len;i++)
   vector(i) = vector(i)/sqrt(norm);
   
  return vector;
}

void printmat(double **mat, int dim)
{
  int i,j;
  for(i=0;i<dim;i++)
    {
      for(j=0;j<dim;j++)
        cout << setprecision(6) << mat[i][j] << "\t";
      cout << "\n" ;
    }
}

int nchoosek(int n, int k)
{
 int r;
 float nck;
 if(2*k>n)
   k = n-k;
   
 for(nck=1.0, r=0;r<k;r++)
 {
  nck = nck*(n-r)/(r+1) ; 
 }
 return (int)nck;
}

