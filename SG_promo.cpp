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
#include "functions.h"

using namespace std; 
using namespace Eigen;
using namespace boost::multiprecision;

#define root2 1.414213562373095
#define PI 3.141592653589793238462


int main(int argc, char** argv)
{
  cout << " This program generates 2 particle states from 1-particle eigenstates " << endl;
  cout << " Format is L,Niter=1, S= -1.0" << endl;
  
  int L,Niter;
  float S;
  
  switch(argc)
  {
   case 4:
     { L = atoi(argv[1]); Niter=atoi(argv[2]); S = atof(argv[3]) ; break; }
   case 3:
     { L = atoi(argv[1]); Niter=atoi(argv[2]); S = -1.0; break; }
   case 2:
     { L = atoi(argv[1]); Niter=1; S=-1.0; break; }  
   default:
      cout << " Input values for atleast L" << endl; return 0;
  }
  
  if(L>=128)
    { cout << "L cannot exceed 127. Program exited" << endl; return 1;} // 127 if basis are integers and 128 if basis are unsigned integers
      
  /* For random number generation */
  struct timeval start;
  struct timeval end;
  // double t1,t2,t3 ,t4;
  time_t seconds, t1, t2;
  time(&seconds);
  srand((unsigned int)seconds);
  
  /* Function declarations */
//  double **symm_gauss_rand_mat_gen(int);
//  MatrixXd hamiltonian(double**, int*,int,int,int);
  int nchoosek(int,int);
  int128_t* definite_particle_basis(int,int);
  double PR(VectorXd);
//  VectorXd gauss_rand_vec_gen(int,int);
 
  /* Variable declarations */
  int i,j,k,l,N,LC2 = nchoosek(L,2),m=1;
  int128_t *basis2, *basis1;
  double norm2, ln3, pr,sigmaJ, **J;
  
  VectorXd vector1, vector2;
  VectorXd promo_vector = VectorXd::Zero(LC2);
  double *q2_entanglement = new double[5]; double *q3_entanglement = new double[2];
  VectorXd vector, promo;
  MatrixXd H;
  
  ostringstream strL, strm, strM, strS;
  string filename;
  strL << L; strS << S;
  
  /* File opening */
  filename = "Data_SG/SG_L";
  // strm.str(std::string());
  // strm << m; 
  filename += strL.str();
  /*
  filename += "_S";
  filename += strS.str();
  filename += "_M";
  filename += strM.str();
  filename += "_m";
  filename += strm.str();
  */
  filename += ".txt";
  
  ofstream myfile, myfile_promo; /* technically, a memory leak */ 
  myfile.open(filename.c_str(),fstream::out | fstream::app);
  
  filename = "Data_SG/SG_promo_L";
  filename += strL.str();
  filename += ".txt";
  myfile_promo.open(filename.c_str(),fstream::out | fstream::app);
  
  if(myfile.is_open())
      cout << "File opened \n";
  else
      cout << "File not opened properly \n";	  
          
    
  /* Basis generation */
  basis2 = definite_particle_basis(2,LC2);
  basis1 = definite_particle_basis(1,L);
  
  for(N=0;N<Niter;N++)
  { 
    cout << "N = " << N << "/" << Niter << endl;
    
    J = symm_gauss_rand_mat_gen(L,S);
   
    sigmaJ = 0;
    for(i=0;i<L;i++)
     for(j=0;j<i;j++)
       sigmaJ += J[i][j];
       
   H = hamiltonian(J,basis1,1,L,L);
   cout << " J and H created \n" ;   
   
   time(&t1);  
   SelfAdjointEigenSolver<MatrixXd> ES(H);
   time(&t2);
   cout << " H diagonalized in " << difftime(t2,t1) << " seconds \n";
   
   // #pragma omp parallel for
   for(l=0;l<L;l++)
   {
    cout << " N = " << N << " l/L = " << l << "/" << L << endl;
    vector1 = ES.eigenvectors().col(l);
    if(abs(vector1.sum())>-0.5) //  avoids the all-one state
    {
     for(k=0,i=1;i<L;i++)
      for(j=0;j<i;j++)
      {
       promo_vector(k++) = vector1(i)+vector1(j);
      }
      
     promo_vector = promo_vector/promo_vector.norm();        
     
     time(&t1); 
     average_concurrence(q2_entanglement, vector1,basis1,L,1,L);  
     time(&t2); // cout << " Average concurrence for 1-particle vector computed in " << difftime(t2,t1) << " seconds \n";
     //LN3(q3_entanglement, vector2,basis2,L,m,LCm);
     q3_entanglement[0] = -1; q3_entanglement[1] = -1;
     pr = PR(vector1);
        
     myfile << pr << "\t" << ES.eigenvalues()[l] << "\t" << ES.eigenvalues()[l] - sigmaJ << "\t" << q2_entanglement[0] << "\t" << q2_entanglement[1] << "\t" << q2_entanglement[2] << "\t" << q2_entanglement[3] << "\t" << q2_entanglement[4] << "\t" << q3_entanglement[0] << "\t" << q3_entanglement[1] << endl;

     time(&t1);
     average_concurrence(q2_entanglement, promo_vector,basis2,L,2,LC2);  
     time(&t2);    // cout << "Average concurrence for the promoted state computed in " << difftime(t2,t1) << " seconds " << endl;
     //LN3(q3_entanglement, promo_vector,basis2,L,m,LCm);
     pr = PR(promo_vector);

     myfile_promo << pr << "\t" << ES.eigenvalues()[l] << "\t" << ES.eigenvalues()[l] - sigmaJ << "\t" << q2_entanglement[0] << "\t" << q2_entanglement[1] << "\t" << q2_entanglement[2] << "\t" << q2_entanglement[3] << "\t" << q2_entanglement[4] << "\t" << q3_entanglement[0] << "\t" << q3_entanglement[1] << endl;
         
    }
    }
   }

  myfile.close();
  myfile_promo.close();
     
  delete basis2; delete basis1;

  /* Niter M*/
  
  cout << '\a';
}

