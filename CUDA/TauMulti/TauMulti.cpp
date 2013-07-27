//////////////////////////////////////////////////////////////////////////
//
//	TauMulti
//		This program is for finding Cij(d) where
//	Cij(d) = \Sum_t Tr[ I_i(t) . T(t, t+d) . I_j(t) . T^{+}(t, t+d)
//
//////////////////////////////////////////////////////////////////////////


//Includes
#include "TauMulti.h"

class dbArguments
{
public:
dbArguments(int tArg, int dArg, cuFloatComplex *hArrArg, char typeArg)
	{
	t = tArg;
	d = dArg;
	hArr = hArrArg;
	type = typeArg;
	}

int t, d;
cuFloatComplex *hArr;
char type;
};

void* setDatabase(void* dbArgumentsVoid);
void* setDatabase_I(void* dbArgumentsVoid);
void* setDatabase_T(void* dbArgumentsVoid);

void* threadTest(void* dbArgumentsVoid);

void testDB_I(int t,int d,cuFloatComplex *hI);
void testDB_T(int t,int d,cuFloatComplex *hT);

void printDB_I(int t,int d,cuFloatComplex *hI);
void printDB_T(int t,int d,cuFloatComplex *hT);





/************************
	Begin Main
************************/

int main()
{





/************************
	Variables
************************/

float MemTotal=0;
int t=0, d=0;

#if ((defined PRINT_C_MATRIX) || (defined DEBUG))
int i;
#endif
//cuFloatComplex cuComplex_tmp;

cuFloatComplex *hI,*hT,*hC; //host
cuFloatComplex *I,*T,*X,*C; //device

dim3 multiBlockDim(MULTI_BLOCK_DIM_X,MULTI_BLOCK_DIM_Y,MULTI_BLOCK_DIM_Z);
dim3 multiGridDim(MULTI_GRID_DIM_X,MULTI_GRID_DIM_Y,1);

dim3 traceBlockDim(TRACE_BLOCK_DIM_X,TRACE_BLOCK_DIM_Y,TRACE_BLOCK_DIM_Z);
dim3 traceGridDim(TRACE_GRID_DIM_X,TRACE_GRID_DIM_Y,1);

dim3 initializeBlockDim_C(INITIALIZE_BLOCK_DIM_C);
dim3 initializeGridDim_C(INITIALIZE_GRID_DIM_C);

dbArguments *dbArgs = (dbArguments*) malloc(sizeof(dbArguments*));

pthread_t dbThread;
int dbThreadReturn = -1;



/*******************************************************************

	Mallocs Section.
		This is a slightly awkward section. There's
		no particularly nice way of putting this that
		I can think of, so hopefully this isn't too
		ugly and indecipherable as it is.
		
		This is used to allocate the necessary space
		on the Device and the Host.

*******************************************************************/

//Set CxC(xd)-dim C(d)
hC = new cuFloatComplex
	[
	C_MATRIX_DIM*C_MATRIX_DIM
	*TIME_LEN
	];

cudaMalloc((void**)&C,
	C_MATRIX_DIM*C_MATRIX_DIM
	*TIME_LEN
	*sizeof(cuFloatComplex)
	);

std::cout << "C: " << printSize(C_MATRIX_DIM*C_MATRIX_DIM*TIME_LEN*sizeof(cuFloatComplex)) << std::endl;


//Set IxI(xC)-dim I_i(t)
hI = new cuFloatComplex
	[
	PHI_MATRIX_DIM*PHI_MATRIX_DIM
	*C_MATRIX_DIM
	*T_CHUNKSIZE
	];

cudaMalloc((void**)&I,
	PHI_MATRIX_DIM*PHI_MATRIX_DIM
	*C_MATRIX_DIM
	*T_CHUNKSIZE
	*sizeof(cuFloatComplex)
	);

std::cout << "I: " << printSize(PHI_MATRIX_DIM*PHI_MATRIX_DIM*C_MATRIX_DIM*T_CHUNKSIZE*sizeof(cuFloatComplex)) << std::endl;


//Set IxI(xtxd)-dim T(t,t+d)
hT = new cuFloatComplex
	[
	PHI_MATRIX_DIM*PHI_MATRIX_DIM
	*T_CHUNKSIZE
	*DELTA_CHUNKSIZE
	];

cudaMalloc((void**)&T,
	PHI_MATRIX_DIM*PHI_MATRIX_DIM
	*T_CHUNKSIZE
	*DELTA_CHUNKSIZE
	*sizeof(cuFloatComplex)
	);

std::cout << "T: " << printSize(PHI_MATRIX_DIM*PHI_MATRIX_DIM*T_CHUNKSIZE*DELTA_CHUNKSIZE*sizeof(cuFloatComplex)) << std::endl;

//Set IxI(xtxdxCx2)-dim X_i^h(t,d)
cudaMalloc((void**)&X,
	PHI_MATRIX_DIM*PHI_MATRIX_DIM
	*2
	*C_MATRIX_DIM
	*T_CHUNKSIZE
	*DELTA_CHUNKSIZE
	*sizeof(cuFloatComplex)
	);

std::cout << "X: " << printSize(PHI_MATRIX_DIM*PHI_MATRIX_DIM*2*C_MATRIX_DIM*T_CHUNKSIZE*DELTA_CHUNKSIZE*sizeof(cuFloatComplex)) << std::endl;


MemTotal += C_MATRIX_DIM*C_MATRIX_DIM*TIME_LEN*sizeof(cuFloatComplex); //C
MemTotal += PHI_MATRIX_DIM*PHI_MATRIX_DIM*C_MATRIX_DIM*T_CHUNKSIZE*sizeof(cuFloatComplex); //I
MemTotal += PHI_MATRIX_DIM*PHI_MATRIX_DIM*T_CHUNKSIZE*DELTA_CHUNKSIZE*sizeof(cuFloatComplex); //T

//Print out the memory used on the HOST
std::cout << std::endl;

std::cout << "Host MemTotal: " << printSize(MemTotal) << std::endl;

//Print out the memory used on the CARD
MemTotal += PHI_MATRIX_DIM*PHI_MATRIX_DIM*2*C_MATRIX_DIM*T_CHUNKSIZE*DELTA_CHUNKSIZE*sizeof(cuFloatComplex); //X

std::cout << "Device MemTotal: " << printSize(MemTotal) << std::endl;
std::cout << std::endl << std::endl;





/***********************
	Main Loops	
***********************/


//Initialize the C array to 0.
#ifdef DEBUG
std::cout << "Initializing C Matrix" << std::endl << std::endl;
#endif

cudaMemset(C, 0, C_MATRIX_DIM*C_MATRIX_DIM*TIME_LEN*sizeof(cuFloatComplex));


//looping type chosen on compile, depending on given options.
#ifdef T_OUTER_LOOP //T_OUTER_LOOP
for(t=0; t<TIME_LEN; t+=T_CHUNKSIZE)
{
#ifdef PRINT_SHORT_LOAD_PROGRESS
std::cout << std::setw(3) << "\r(" << t << "," << d << ")" << std::flush;
#endif

//Initial database call
#ifdef DEBUG
std::cout << "Initial Load" << std::endl << std::endl;
#endif

#if defined DEBUG || defined PRINT_FUNC_RETURNS
std::cout << "Running Initial setDatabase" << std::endl;
#endif

*dbArgs = dbArguments(t, 0, hI, 'I');
setDatabase_I( static_cast<void*>(dbArgs) );

*dbArgs = dbArguments(t, 0, hT, 'T');
setDatabase_T( static_cast<void*>(dbArgs) );

#if defined DEBUG || defined PRINT_FUNC_RETURNS
std::cout << std::endl;
#endif

#if ((defined PRINT_LOADS) || (defined DEBUG)) && (!defined SUPPRESS_PRINT_LOADS)
printDB_I(t,d,hI);
#endif
cudaMemcpy(I, hI, PHI_MATRIX_DIM*PHI_MATRIX_DIM*C_MATRIX_DIM*T_CHUNKSIZE*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

	for(d=0; d<TIME_LEN; d+=DELTA_CHUNKSIZE)
	{
	//END T_OUTER LOOP
	
#else //D_OUTER_LOOP
for(d=0; d<TIME_LEN; d+=DELTA_CHUNKSIZE)
{
#ifdef PRINT_SHORT_LOAD_PROGRESS
std::cout << "\r(" << t << "," << d << ")" << std::flush;
#endif
	
//Initial database call
#ifdef DEBUG
std::cout << "Initial Load" << std::endl;
#endif

#if defined DEBUG || defined PRINT_FUNC_RETURNS
std::cout << "Running Initial setDatabase" << std::endl;
#endif

*dbArgs = dbArguments(0, d, hI, 'I');
setDatabase_I( static_cast<void*>(dbArgs) );

*dbArgs = dbArguments(0, d, hT, 'T');
setDatabase_T( static_cast<void*>(dbArgs) );

#if defined DEBUG || defined PRINT_FUNC_RETURNS
std::cout << std::endl;
#endif

	for(t=0; t<TIME_LEN; t+=T_CHUNKSIZE)
	{
	#if ((defined PRINT_LOADS) || (defined DEBUG)) && (!defined SUPPRESS_PRINT_LOADS)
	printDB_I(t,d,hI);
	#endif
	cudaMemcpy(I, hI, PHI_MATRIX_DIM*PHI_MATRIX_DIM*C_MATRIX_DIM*T_CHUNKSIZE*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
#endif //END D_OUTER_LOOP

	
	//Upload matrices from HOST to CARD
	#if ((defined PRINT_LOADS) || (defined DEBUG)) && (!defined SUPPRESS_PRINT_LOADS)
	printDB_T(t,d,hT);
	#endif
	cudaMemcpy(T, hT, PHI_MATRIX_DIM*PHI_MATRIX_DIM*T_CHUNKSIZE*DELTA_CHUNKSIZE*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
	
	//next set of matrices loaded in parallel thread from database
	#ifdef T_OUTER_LOOP
	if(d+DELTA_CHUNKSIZE<TIME_LEN)
		{
		#ifdef DEBUG
		std::cout << "Preload Load" << std::endl;
		std::cout << "time: (" << t << "," << d << ")" << std::endl;
		std::cout << "load start: (" << t << "," << d + DELTA_CHUNKSIZE << ")" << std::endl << std::endl;
		#endif
		
		#if defined DEBUG || defined PRINT_FUNC_RETURNS
		std::cout << "Running setDatabase" << std::endl;
		#endif
		
		*dbArgs = dbArguments(t, d + DELTA_CHUNKSIZE, hT, 'T');
		#ifdef DATABASE_SERIAL
		setDatabase_T(dbArgs);
		#else
		dbThreadReturn = pthread_create(&dbThread, NULL, &setDatabase_T, (void*) dbArgs);
		#endif
		}
	#else
	if(t+T_CHUNKSIZE<TIME_LEN)
		{
		#ifdef DEBUG
		std::cout << "Preload Load" << std::endl;
		std::cout << "time: (" << t << "," << d << ")" << std::endl;
		std::cout << "load start: (" << t + T_CHUNKSIZE << "," << d  << ")" << std::endl << std::endl;
		#endif
		
		#if defined DEBUG || defined PRINT_FUNC_RETURNS
		std::cout << "Running setDatabase" << std::endl;
		#endif
		
		*dbArgs = dbArguments(t + T_CHUNKSIZE, d, hT, 'T');
		#ifdef DATABASE_SERIAL
		setDatabase_T(dbArgs);
		#else
		dbThreadReturn = pthread_create(&dbThread, NULL, &setDatabase_T, static_cast<void*>(dbArgs) );
		#endif
		}
	#endif
	
	#if defined DEBUG || defined PRINT_FUNC_RETURNS
	std::cout << "Running Kernels" << std::endl;
	#endif
	
	Multiply(multiGridDim, multiBlockDim, I, T, X);
	
	#if defined DEBUG || defined PRINT_FUNC_RETURNS
	std::cout << "Multiply Returned" << std::endl;
	#endif
	
	Trace(traceGridDim, traceBlockDim, C, X, d);
	
	#if defined DEBUG || defined PRINT_FUNC_RETURNS
	std::cout << "Trace Returned" << std::endl;
	#endif
	
	//Sync back with the database thread
	if(dbThreadReturn==0)
		{
		pthread_join(dbThread,NULL);
		}
	
	#if defined DEBUG || defined PRINT_FUNC_RETURNS
	std::cout << std::endl;
	#endif
	}
}
//main loop done


//Download matrices from CARD to HOST
cudaMemcpy(hC, C, C_MATRIX_DIM*C_MATRIX_DIM*TIME_LEN*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);


#if ((defined PRINT_C_MATRIX) || (defined DEBUG))
//Print C Matrix
std::cout << std::endl << "C:";
for(i=0; i<TIME_LEN*C_MATRIX_DIM*C_MATRIX_DIM; i++)
	{
	if(i%(C_MATRIX_DIM) == 0)
		{
		std::cout  << std::endl;
		}
	if(i%(C_MATRIX_DIM*C_MATRIX_DIM) == 0)
		{
		int d_val = static_cast<int>( i/(C_MATRIX_DIM*C_MATRIX_DIM) );
		std::cout  << std::endl << "d = " << d_val << std::endl;
		}
	std::cout  << cuComplexftoString(hC[i]) << " ";
	}
std::cout << std::endl << std::endl;
#endif

#ifdef PULL_X
cudaMemcpy(hI, X + PULL_X_POS, PHI_MATRIX_DIM*PHI_MATRIX_DIM*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
//Print C Matrix
std::cout << std::endl << "X:";
std::cout << std::endl << "X size: " << PHI_MATRIX_DIM*PHI_MATRIX_DIM*2*C_MATRIX_DIM*T_CHUNKSIZE*DELTA_CHUNKSIZE
	<< std::endl << "Pull_X_pos: " << PULL_X_POS;
for(i=0; i<PHI_MATRIX_DIM*PHI_MATRIX_DIM; i++)
	{
	if(i%(PHI_MATRIX_DIM) == 0)
		{
		std::cout  << std::endl;
		}
	if(i%(PHI_MATRIX_DIM*PHI_MATRIX_DIM) == 0)
		{
		std::cout  << std::endl << "h = " << PULL_X_H << std::endl
			<< "i = " << PULL_X_I << std::endl
			<< "t = (" << PULL_X_T << "," << PULL_X_D << ")" << std::endl;
		}
	std::cout  << cuComplexftoString(hI[i + PULL_X_POS]) << " ";
	}
std::cout << std::endl << std::endl;
#endif



/*******************************
	Clean Up and Exit	
*******************************/

//Free a bunch of space
#ifdef DEBUG
std::cout << "Freeing CUDA pointers" << std::endl;
#endif
cudaFree(C);
cudaFree(I);
cudaFree(T);
cudaFree(X);
#ifdef DEBUG
std::cout << "Freeing HOST pointers" << std::endl << std::endl;
#endif
delete hC; //seems to give double free or corruption error...
delete hI;
delete hT;
free(dbArgs);

return NULL;
}


//silly function to see if threads are actually working
void* threadTest(void* integer)
{
int i;
int args = *static_cast<int*>(integer);
for(i=0; i<100; i++)
	{
	std::cout << "threaded: " << args << std::endl;
	}
return NULL;
}





/********************************
	extra functions
********************************/





/*******************************************************************************

	setDatabase
		This is a database wrapper thatis called in a separate thread.
		The reason for the separation is the slightly awkward syntax of
		pthread's thread creation API, requiring that only one
		argument be passed to the function, so multiple arguments
		should be passed wrapped in a struct. In order to avoid
		other database functions be wrapped in a struct, the
		struct is pulled apart here, and the given database function
		is called appropriately here.

*******************************************************************************/

void* setDatabase(void* dbArgumentsVoid)
{
dbArguments dbArgs = *static_cast<dbArguments*>(dbArgumentsVoid);

int t = dbArgs.t;
int d = dbArgs.d;
cuFloatComplex *hArr;
hArr = dbArgs.hArr;
char type = dbArgs.type;

if(type=='I')
	{
	testDB_I(t, d, hArr);
	}

if(type=='T')
	{
	testDB_T(t, d, hArr);
	}

#if defined DEBUG || defined PRINT_FUNC_RETURNS
std::cout << "DB returned" << std::endl;
#endif

return NULL;
}

void* setDatabase_I(void* dbArgumentsVoid)
{
dbArguments dbArgs = *static_cast<dbArguments*>(dbArgumentsVoid);

testDB_I(dbArgs.t, dbArgs.d, dbArgs.hArr);

#if defined DEBUG || defined PRINT_FUNC_RETURNS
std::cout << "DB I returned" << std::endl;
#endif

return NULL;
}

void* setDatabase_T(void* dbArgumentsVoid)
{
dbArguments dbArgs = *static_cast<dbArguments*>(dbArgumentsVoid);

testDB_T(dbArgs.t, dbArgs.d, dbArgs.hArr);

#if defined DEBUG || defined PRINT_FUNC_RETURNS
std::cout << "DB T returned" << std::endl;
#endif

return 0;
}

void printDB_I(int t, int d, cuFloatComplex *hI)
{
int k;

std::cout << std::endl  << "Loading:" << std::endl;
std::cout << std::endl << "PHI:";
for(k=0; k<C_MATRIX_DIM*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM; k++)
	{
	if(k%(PHI_MATRIX_DIM) == 0)
		{
		std::cout  << std::endl;
		}
	if(k%(PHI_MATRIX_DIM*PHI_MATRIX_DIM) == 0)
		{
		int i_val = static_cast<int>( k/(T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM) );
		int t_val = static_cast<int>(
					(k - i_val*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM)
					/(PHI_MATRIX_DIM*PHI_MATRIX_DIM)
				);
		std::cout  << std::endl << "i = " << i_val << std::endl
			<< "t = " << t + t_val << std::endl;
		}
	std::cout  << cuComplexftoString(hI[k]) << " ";
	}
std::cout << std::endl << "*****" << std::endl;
}

void printDB_T(int t, int d, cuFloatComplex *hT)
{
int k;

std::cout << std::endl << "T:";
for(k=0; k<DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM; k++)
	{
	if(k%(PHI_MATRIX_DIM) == 0)
		{
		std::cout  << std::endl;
		}
	if(k%(PHI_MATRIX_DIM*PHI_MATRIX_DIM) == 0)
		{
		int d_val = static_cast<int>( k/(T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM) );
		int t_val = static_cast<int>(
					(k - d_val*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM)
					/(PHI_MATRIX_DIM*PHI_MATRIX_DIM)
				);
		std::cout  << std::endl << "t = (" << t + t_val << "," << d + d_val << ")" << std::endl;
		}
	std::cout  << cuComplexftoString(hT[k]) << " ";
	}
std::cout << std::endl << "*****" << std::endl << std::endl << std::endl;
}


/*******************************************************************************

	testDB
		This is a dumb database function, returning matrices that
		are very easily manipulated by hand (or by head).
		As of writing, diagonal matrices, with entries based on
		the given time and delta are returned, with a 1 in the top
		right hand corner. The results of these are very obvious, as
		the diagonals of the resulting matrix should be the product
		of the two given diagonal entries, and the top right entry
		should be the sum of the diagonal entries.
		
*******************************************************************************/

void testDB_I(int t, int d, cuFloatComplex *hI)
{
int i_it;

int t_it;
int t_val;

int k_it;
int k_val;

//set I
for(i_it=0; i_it<C_MATRIX_DIM; i_it++)
	{
	for(t_it=0; t_it<T_CHUNKSIZE; t_it++)
		{
		t_val = t + t_it;

		for(k_it=0; k_it<PHI_MATRIX_DIM*PHI_MATRIX_DIM; k_it++)
			{
			k_val = i_it*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
				t_it*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
				k_it;
		
		
			if(k_it%(PHI_MATRIX_DIM + 1) == 0)
				{
				#ifdef I_UNIT_MATRIX
				hI[k_val] = make_cuFloatComplex(1,0);
				#elif defined I_USUAL_MATRIX
				hI[k_val] = make_cuFloatComplex((float)3*(t_val+1),(float)(i_it));
				#else
				hI[k_val] = make_cuFloatComplex(1,0);
				#endif
				}
			else
				{
				hI[k_val] = make_cuFloatComplex(0,0);
			
				#ifdef I_USUAL_MATRIX
				if(k_it==(PHI_MATRIX_DIM-1))
					{
					hI[k_val] = make_cuFloatComplex(1,0);
					}
				#endif
				}
			}
		}
	}
}

/**********
Profiler says a long time is spent with make_cuFloatComplex.
**********/

/*void testDB_I(int t, int d, cuFloatComplex *hI)
{
int i_it;

int t_it;
int t_val;

int k_it;
int k_val;

float *hIf;

hIf = (float*) hI;

//set I
for(i_it=0; i_it<C_MATRIX_DIM; i_it++)
	{
	for(t_it=0; t_it<T_CHUNKSIZE; t_it++)
		{
		t_val = t + t_it;

		for(k_it=0; k_it<2*PHI_MATRIX_DIM*PHI_MATRIX_DIM; k_it+=2)
			{
			k_val = i_it*T_CHUNKSIZE*2*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
				t_it*2*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
				k_it;
		
		
			if(k_it%(2*PHI_MATRIX_DIM + 2) == 0)
				{
				#ifndef I_UNIT_MATRIX
				hIf[k_val] = 3*(t_val+1);
				hIf[k_val+1] = i_it;
				#else
				hIf[k_val] = 1;
				hIf[k_val+1] = 0
				#endif
				}
			else
				{
				hIf[k_val] = 0;
				hIf[k_val+1] = 0;
			
				#ifndef I_UNIT_MATRIX
				if(k_it==(2*PHI_MATRIX_DIM-2))
					{
					hIf[k_val] = 1;
					hIf[k_val+1] = 0;
					}
				#endif
				}
			}
		}
	}
}*/





void testDB_T(int t, int d, cuFloatComplex *hT)
{
int t_it;
int t_val;

int d_it;
int d_val;

int k_it;
int k_val;

//set T
for(d_it=0; d_it<DELTA_CHUNKSIZE; d_it++)
	{
	d_val = d + d_it;
	
	for(t_it=0; t_it<T_CHUNKSIZE; t_it++)
		{
		t_val = t + t_it;
	
		for(k_it=0; k_it<PHI_MATRIX_DIM*PHI_MATRIX_DIM; k_it++)
			{
			k_val = d_it*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
				t_it*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
				k_it;
		
		
			if(k_it%(PHI_MATRIX_DIM + 1) == 0)
				{
				#ifdef T_UNIT_MATRIX
				hT[k_val] = make_cuFloatComplex(1,0);
				#elif defined T_USUAL_MATRIX
				hT[k_val] = make_cuFloatComplex((float)2*(t_val+1),(float)(d_val+1));
				#else
				hT[k_val] = make_cuFloatComplex(1,0);
				#endif
				}
			else
				{
				hT[k_val] = make_cuFloatComplex(0,0);
				
				#ifdef T_USUAL_MATRIX
				if(k_it==(PHI_MATRIX_DIM-1))
					{
					hT[k_val] = make_cuFloatComplex(1,0);
					}
				#endif
				}
			}
		}
	}
}


/*void testDB_T(int t, int d, cuFloatComplex *hT)
{
int t_it;
int t_val;

int d_it;
int d_val;

int k_it;
int k_val;

float *hTf;
hTf = (float*) hT;

//set T
for(d_it=0; d_it<DELTA_CHUNKSIZE; d_it++)
	{
	d_val = d + d_it;
	
	for(t_it=0; t_it<T_CHUNKSIZE; t_it++)
		{
		t_val = t + t_it;
	
		for(k_it=0; k_it<2*PHI_MATRIX_DIM*PHI_MATRIX_DIM; k_it+=2)
			{
			k_val = d_it*T_CHUNKSIZE*2*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
				t_it*2*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
				k_it;
		
		
			if(k_it%(2*PHI_MATRIX_DIM + 2) == 0)
				{
				#ifdef T_UNIT_MATRIX
				hTf[k_val] = 1;
				hTf[k_val+1] = 0;
				#else
				hTf[k_val] = 2*(t_val+1);
				hTf[k_val+1] = d_val+1;
				#endif
				}
			else
				{
				hTf[k_val] = 0;
				hTf[k_val+1] = 0;
				
				#ifndef T_UNIT_MATRIX
				if(k_it==(2*PHI_MATRIX_DIM-2))
					{
					hTf[k_val] = 1;
					hTf[k_val+1] = 0;
					}
				#endif
				}
			}
		}
	}
}*/
