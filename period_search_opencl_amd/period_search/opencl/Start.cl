//#include "GlobalsCL.h"
kernel void ClCheckEnd(
	__global int* CUDA_End,
	int theEnd)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);

	if (blockIdx.x == 0)
		*CUDA_End = theEnd;

	//if (blockIdx.x == 0)
		//printf("CheckEnd CUDA_End: %2d\n", *CUDA_End);

}

__kernel void ClCalculatePrepare(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_result* CUDA_FR,
	__global int* CUDA_End,
	double freq_start,
	double freq_step,
	int n_max,
	int n_start)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);
	int x = blockIdx.x;

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	int n = n_start + blockIdx.x;


	//zero context
	if (n > n_max)
	{
		//CUDA_mCC[x].isInvalid = 1;
		(*CUDA_LCC).isInvalid = 1;
		(*CUDA_FR).isInvalid = 1;
		return;
	}
	else
	{
		//CUDA_mCC[x].isInvalid = 0;
		(*CUDA_LCC).isInvalid = 0;
		(*CUDA_FR).isInvalid = 0;
	}

	//printf("[%d] n_start: %d | n_max: %d | n: %d \n", blockIdx.x, n_start, n_max, n);

	//printf("Idx: %d | isInvalid: %d\n", x, CUDA_CC[x].isInvalid);
	//printf("Idx: %d | isInvalid: %d\n", x, (*CUDA_LCC).isInvalid);

	//CUDA_mCC[x].freq = freq_start - (n - 1) * freq_step;
	(*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;

	///* initial poles */
	(*CUDA_LFR).per_best = 0.0;
	(*CUDA_LFR).dark_best = 0.0;
	(*CUDA_LFR).la_best = 0.0;
	(*CUDA_LFR).be_best = 0.0;
	(*CUDA_LFR).dev_best = 1e40;

	//printf("n: %4d, CUDA_CC[%3d].freq: %10.7f, CUDA_FR[%3d].la_best: %10.7f, isInvalid: %4d \n", n, x, (*CUDA_LCC).freq, x, (*CUDA_LFR).la_best, (*CUDA_LCC).isInvalid);

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	//if (blockIdx.x == 0)
		//printf("Prepare CUDA_End: %2d\n", *CUDA_End);
}

__kernel void ClCalculatePreparePole(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__global double* CUDA_cg_first,
	__global int* CUDA_End,
	__global struct freq_context* CUDA_CC2,
	//double CUDA_cl,
	int m)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);
	int x = blockIdx.x;

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	//const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	//int t = *CUDA_End;
	//*CUDA_End = 13;
	//printf("[%d] PreparePole t: %d, CUDA_End: %d\n", x, t, *CUDA_End);


	if ((*CUDA_LCC).isInvalid)
	{
		//atomic_add(CUDA_End, 1);
		atomic_inc(CUDA_End);
		//printf("prepare pole %d ", (*CUDA_End));

		(*CUDA_FR).isReported = 0; //signal not to read result

		//printf("[%d] isReported: %d \n", blockIdx.x, (*CUDA_FR).isReported);

		return;
	}

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("[Device] PreparePole > ma: %d\n", (*CUDA_CC).ma);

	double period = 1.0 / (*CUDA_LCC).freq;

	//* starts from the initial ellipsoid */
	for (int i = 1; i <= (*CUDA_CC).Ncoef; i++)
	{
		(*CUDA_LCC).cg[i] = CUDA_cg_first[i];
		//if(blockIdx.x == 0)
		//	printf("cg[%3d]: %10.7f\n", i, CUDA_cg_first[i]);
	}
	//printf("Idx: %d | m: %d | Ncoef: %d\n", x, m, (*CUDA_CC).Ncoef);
	//printf("cg[%d]: %.7f\n", x, CUDA_CC[x].cg[CUDA_CC[x].Ncoef + 1]);
	//printf("Idx: %d | beta_pole[%d]: %.7f\n", x, m, CUDA_CC[x].beta_pole[m]);

	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = (*CUDA_CC).beta_pole[m];
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2] = (*CUDA_CC).lambda_pole[m];
	//if (blockIdx.x == 0)
	//{
	//	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);
	//	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);
	//}
	//printf("cg[%d]: %.7f | cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1], (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);

	/* The formulas use beta measured from the pole */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = 90.0 - (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];
	//printf("90 - cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);

	/* conversion of lambda, beta to radians */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = DEG2RAD * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2] = DEG2RAD * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2];
	//printf("cg[%d]: %.7f | cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1], (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);

	/* Use omega instead of period */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3] = 24.0 * 2.0 * PI / period;

	//if (threadIdx.x == 0)
	//{
	//	printf("[%3d] cg[%3d]: %10.7f, period: %10.7f\n", blockIdx.x, (*CUDA_CC).Ncoef + 3, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3], period);
	//}

	for (int i = 1; i <= (*CUDA_CC).Nphpar; i++)
	{
		(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i] = (*CUDA_CC).par[i];
		//              ia[Ncoef+3+i] = ia_par[i]; moved to global
		//if (blockIdx.x == 0)
		//	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + i, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i]);

	}

	/* Lommel-Seeliger part */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2] = 1;
	//if (blockIdx.x == 0)
	//{
	//	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2]);
	//}

	/* Use logarithmic formulation for Lambert to keep it positive */
	(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1] = log((*CUDA_CC).cl);
	//(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1] = (*CUDA_CC).logCl;   //log((*CUDA_CC).cl);


	//if (blockIdx.x == 0)
	//{
	//	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1]);
	//}
	//printf("cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1]);

	/* Levenberg-Marquardt loop */
	// moved to global iter_max,iter_min,iter_dif_max
	//
	(*CUDA_LCC).rchisq = -1;
	(*CUDA_LCC).Alamda = -1;
	(*CUDA_LCC).Niter = 0;
	(*CUDA_LCC).iter_diff = 1e40;
	(*CUDA_LCC).dev_old = 1e30;
	(*CUDA_LCC).dev_new = 0;
	//	(*CUDA_LCC).Lastcall=0; always ==0
	(*CUDA_LFR).isReported = 0;

	// TODO: Check if this is needed at all!!!
	//if (blockIdx.x == 0)
	//{
	//    for (int i = 0; i < MAX_N_OBS + 1; i++)
	//    {
	//        //printf("[%d] %g", blockIdx.x, (*CUDA_CC).Brightness[i]);
	//        (*CUDA_CC2).Brightness[i] = (*CUDA_CC).Brightness[i];
	//    }
	//}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Begin(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_result* CUDA_FR,
	__global int* CUDA_End,
	int CUDA_n_iter_min,
	int CUDA_n_iter_max,
	double CUDA_iter_diff_max,
	double CUDA_Alamda_start)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);
	int x = blockIdx.x;

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	//const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	//                                   ?    < 50                                 ?       > 0                                   ?      < 0
	(*CUDA_LCC).isNiter = (((*CUDA_LCC).Niter < CUDA_n_iter_max) && ((*CUDA_LCC).iter_diff > CUDA_iter_diff_max)) || ((*CUDA_LCC).Niter < CUDA_n_iter_min);
	(*CUDA_FR).isNiter = (*CUDA_LCC).isNiter;

	//printf("[%d] isNiter: %d, Alamda: %10.7f\n", blockIdx.x, (*CUDA_LCC).isNiter, (*CUDA_LCC).Alamda);

	if ((*CUDA_LCC).isNiter)
	{
		if ((*CUDA_LCC).Alamda < 0)
		{
			(*CUDA_LCC).isAlamda = 1;
			(*CUDA_LCC).Alamda = CUDA_Alamda_start; /* initial alambda */
		}
		else
		{
			(*CUDA_LCC).isAlamda = 0;
		}
	}
	else
	{
		if (!(*CUDA_LFR).isReported)
		{
			//int oldEnd = *CUDA_End;
			//atomic_add(CUDA_End, 1);
			int t = *CUDA_End;
			atomic_inc(CUDA_End);

			//printf("[%d] t: %2d, Begin %2d\n", blockIdx.x, t, *CUDA_End);

			(*CUDA_LFR).isReported = 1;
		}
	}

	//if (threadIdx.x == 1)
	//	printf("[begin] Alamda: %10.7f\n", (*CUDA_LCC).Alamda);
	//barrier(CLK_GLOBAL_MEM_FENCE); // TEST
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Start(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
	//__global int* CUDA_End)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);
	int x = blockIdx.x;

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	//double* dytemp = &CUDA_Dytemp[blockIdx.x];

	//double* Area = &CUDA_mCC[0].Area;

	//if (blockIdx.x == 0)
	//	printf("[%d][%3d] [Mrqcof1Start]\n", blockIdx.x, threadIdx.x);
		//printf("isInvalid: %3d, isNiter: %3d, isAlamda: %3d\n", (*CUDA_LCC).isInvalid, (*CUDA_LCC).isNiter, (*CUDA_LCC).isAlamda);

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return; //>> 0

	// => mrqcof_start(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
	mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Matrix(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global double* bufTim,
	__global double* bufEe,
	__global double* bufEe0,
	__global double* jpScale, // set
	__global double* jpDphp1, // set
	__global double* jpDphp2, // set
	__global double* jpDphp3, // set
	__global double* e1,
	__global double* e2,
	__global double* e3,
	__global double* e01,
	__global double* e02,
	__global double* e03,
	__global double* de,
	__global double* de0,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);
	int x = blockIdx.x;

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	int commonSize = x * ((*CUDA_CC).maxLcPoints + 1);
	int B = ((*CUDA_CC).maxLcPoints + 1);
	int C = 4;
	int D = 4;

	__global double* mJpScale = &jpScale[commonSize];
	__global double* mJpDphp1 = &jpDphp1[commonSize];
	__global double* mJpDphp2 = &jpDphp2[commonSize];
	__global double* mJpDphp3 = &jpDphp3[commonSize];
	__global double* mE1 = &e1[commonSize];
	__global double* mE2 = &e2[commonSize];
	__global double* mE3 = &e3[commonSize];
	__global double* mE01 = &e01[commonSize];
	__global double* mE02 = &e02[commonSize];
	__global double* mE03 = &e03[commonSize];

	// Create a pointer to the slice de[1][][][]
	// double *sDe = &flattened_array[1 * MAX_POINTS * N * N];
	__global double* mDe = &de[x * B * C * D];
	__global double* mDe0 = &de0[x * B * C * D];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	__local int num; // __shared__

	int3 localIdx;
	localIdx.x = get_local_id(0);
	if (localIdx.x == 0)
	{
		num = 0;
	}

	mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, bufTim, bufEe, bufEe0, mJpScale, mJpDphp1, mJpDphp2, mJpDphp3, mE1, mE2, mE3, mE01, mE02, mE03,
		mDe, mDe0, lpoints, num);
	//if (x == 1)
	//{
		// flat_index = (i * B * C * D) + (j * C * D) + (k * D) + l;

		for (int jp = 0; jp < 10; jp++)
		{
			for (int k = 0; k < 4; k++)
			{
				int dX = (x * B * C * D) + (jp * C * D) + (k * D) + 1;
				int mX = (jp * C * D) + (k * D) + 1;
				if(de[dX] != mDe[mX])
				{
					printf("jp[%3d] de[%3d] %8.6f mDe[%3d] %8.6f false\n", jp, dX, de[dX], mX, mDe[mX]);
				}
			}
		}

	//}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Curve1(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global double* jpScale, // get
	__global double* jpDphp1, // set
	__global double* jpDphp2, // set
	__global double* jpDphp3, // set
	__global double* e1,
	__global double* e2,
	__global double* e3,
	__global double* e01,
	__global double* e02,
	__global double* e03,
	__global double* de,
	__global double* de0,
	__global double* bufDytemp,
	__global double* bufYtemp,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);
	int x = blockIdx.x;

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	int commonSize = blockIdx.x * ((*CUDA_CC).maxLcPoints + 1);
	int B = ((*CUDA_CC).maxLcPoints + 1);
	int C = 4;
	int D = 4;

	__global double* mJpScale = &jpScale[commonSize];
	__global double* mJpDphp1 = &jpDphp1[commonSize];
	__global double* mJpDphp2 = &jpDphp2[commonSize];
	__global double* mJpDphp3 = &jpDphp3[commonSize];
	__global double* mE1 = &e1[commonSize];
	__global double* mE2 = &e2[commonSize];
	__global double* mE3 = &e3[commonSize];
	__global double* mE01 = &e01[commonSize];
	__global double* mE02 = &e02[commonSize];
	__global double* mE03 = &e03[commonSize];
	__global double* ytemp = &bufYtemp[commonSize];

	__global double* mDe = &de[x * B * C * D];
	__global double* mDe0 = &de0[x * B * C * D];

	__global double* mDytemp = &bufDytemp[commonSize * (MAX_N_PAR + 1)];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	__local int num;  // __shared__
	__local double tmave[BLOCK_DIM];

	if (threadIdx.x == 0)
	{
		num = 0;
	}

	mrqcof_curve1(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, mJpScale, mJpDphp1, mJpDphp2, mJpDphp3, mE1, mE2, mE3, mE01, mE02, mE03,
		mDe, mDe0, mDytemp, ytemp, tmave, inrel, lpoints, num);

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("[Mrqcof1Curve1] [%d][%3d] alpha[56]: %10.7f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).alpha[56]);

	//if (blockIdx.x == 0)
	//	printf("dytemp[8636]: %10.7f\n", dytemp[8636]);
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Curve1Last(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global double* bufDytemp,
	__global double* bufYtemp,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	int commonSize = blockIdx.x * ((*CUDA_CC).maxLcPoints + 1);
	__global double* mDytemp = &bufDytemp[commonSize * (MAX_N_PAR + 1)];
	__global double* ytemp = &bufYtemp[commonSize];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	__local double res[BLOCK_DIM];

	mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, mDytemp, ytemp, res, inrel, lpoints);
	//if (threadIdx.x == 0)
	//{
	//	int i = 56;
	//	//for (int i = 1; i <= 60; i++) {
	//		printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
	//	//}
	//}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1Curve2(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global double* brightness,
	__global double* weight,
	__global double* sig,
	__global double* bufDytemp,
	__global double* bufYtemp,
	__global int* ia,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	int commonSize = blockIdx.x * ((*CUDA_CC).maxLcPoints + 1);
	__global double* mDytemp = &bufDytemp[commonSize * (MAX_N_PAR + 1)];
	__global double* ytemp = &bufYtemp[commonSize];

	//if (blockIdx.x == 0)
	//printf("[%3d] isInvalid: %3d, isNiter: %3d, isAlamda: %3d\n", threadIdx.x, (*CUDA_LCC).isInvalid, (*CUDA_LCC).isNiter, (*CUDA_LCC).isAlamda);

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve2(CUDA_LCC, CUDA_CC, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, brightness, weight, sig, mDytemp, ytemp, ia, inrel, lpoints);

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("[Mrqcof1Curve2] [%d][%3d] alpha[56]: %10.7f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).alpha[56]);

	//if (threadIdx.x == 0)
	//{
	//	int i = 56;
	//	//for (int i = 1; i <= 60; i++) {
	//	printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
	//	//}
	//}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof1End(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqcof1End\n");


	(*CUDA_LCC).Ochisq = mrqcof_end(CUDA_LCC, CUDA_CC, (*CUDA_LCC).alpha);


	////if (threadIdx.x == 0)
	////{
	//	int i = 56;
	//	//for (int i = 1; i <= 60; i++) {
	//	printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
	//	//}
	////}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqmin1End(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global int* ia)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//if (threadIdx.x == 0)
	//{
	//	int i = 56;
	//	//for (int i = 1; i <= 60; i++)
	//	//{
	//		printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
	//	//}
	//}

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqmin1End\n");

	// gauss_err =
	//mrqmin_1_end(CUDA_LCC, CUDA_CC, sh_icol, sh_irow, sh_big, icol, pivinv);


	mrqmin_1_end(CUDA_LCC, CUDA_CC, ia);

	//if (blockIdx.x == 0) {
	//	printf("[%3d] sh_icol[%3d]: %3d\n", threadIdx.x, threadIdx.x, sh_icol[threadIdx.x]);
	//}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Start(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqcof2Start\n");


	//mrqcof_start(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);
	mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("alpha[56]: %10.7f\n", (*CUDA_LCC).alpha[56]);
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Matrix(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global double* bufTim,
	__global double* bufEe,
	__global double* bufEe0,
	__global double* jpScale,
	__global double* jpDphp1,
	__global double* jpDphp2,
	__global double* jpDphp3,
	__global double* e1,
	__global double* e2,
	__global double* e3,
	__global double* e01,
	__global double* e02,
	__global double* e03,
	__global double* de,
	__global double* de0,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	int x = blockIdx.x;

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	int commonSize = blockIdx.x * ((*CUDA_CC).maxLcPoints + 1);
	int B = ((*CUDA_CC).maxLcPoints + 1);
	int C = 4;
	int D = 4;

	__global double* mJpScale = &jpScale[commonSize];
	__global double* mJpDphp1 = &jpDphp1[commonSize];
	__global double* mJpDphp2 = &jpDphp2[commonSize];
	__global double* mJpDphp3 = &jpDphp3[commonSize];
	__global double* mE1 = &e1[commonSize];
	__global double* mE2 = &e2[commonSize];
	__global double* mE3 = &e3[commonSize];
	__global double* mE01 = &e01[commonSize];
	__global double* mE02 = &e02[commonSize];
	__global double* mE03 = &e03[commonSize];

	__global double* mDe = &de[x * B * C * D];
	__global double* mDe0 = &de0[x * B * C * D];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	__local int num; // __shared__

	int3 localIdx;
	localIdx.x = get_local_id(0);
	if (localIdx.x == 0)
	{
		num = 0;
	}

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqcof2Matrix\n");

	//mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).atry, lpoints);
	mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, bufTim, bufEe, bufEe0, mJpScale, mJpDphp1, mJpDphp2, mJpDphp3, mE1, mE2, mE3, mE01, mE02, mE03,
		mDe, mDe0, lpoints, num);

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Curve1(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global double* jpScale,
	__global double* jpDphp1,
	__global double* jpDphp2,
	__global double* jpDphp3,
	__global double* e1,
	__global double* e2,
	__global double* e3,
	__global double* e01,
	__global double* e02,
	__global double* e03,
	__global double* de,
	__global double* de0,
	__global double* bufDytemp,
	__global double* bufYtemp,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);
	int x = blockIdx.x;

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	int commonSize = blockIdx.x * ((*CUDA_CC).maxLcPoints + 1);
	int B = ((*CUDA_CC).maxLcPoints + 1);
	int C = 4;
	int D = 4;

	__global double* mJpScale = &jpScale[commonSize];
	__global double* mJpDphp1 = &jpDphp1[commonSize];
	__global double* mJpDphp2 = &jpDphp2[commonSize];
	__global double* mJpDphp3 = &jpDphp3[commonSize];
	__global double* mE1 = &e1[commonSize];
	__global double* mE2 = &e2[commonSize];
	__global double* mE3 = &e3[commonSize];
	__global double* mE01 = &e01[commonSize];
	__global double* mE02 = &e02[commonSize];
	__global double* mE03 = &e03[commonSize];
	__global double* ytemp = &bufYtemp[commonSize];

	__global double* mDe = &de[x * B * C * D];
	__global double* mDe0 = &de0[x * B * C * D];

	__global double* mDytemp = &bufDytemp[commonSize * (MAX_N_PAR + 1)];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	__local int num;  // __shared__
	__local double tmave[BLOCK_DIM];

	if (threadIdx.x == 0)
	{
		num = 0;
	}

	mrqcof_curve1(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, mJpScale, mJpDphp1, mJpDphp2, mJpDphp3, mE1, mE2, mE3, mE01, mE02, mE03,
		mDe, mDe0, mDytemp, ytemp, tmave, inrel, lpoints, num);

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Curve2(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global double* brightness,
	__global double* weight,
	__global double* sig,
	__global double* bufDytemp,
	__global double* bufYtemp,
	__global int* ia,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	int commonSize = blockIdx.x * ((*CUDA_CC).maxLcPoints + 1);
	__global double* mDytemp = &bufDytemp[commonSize * (MAX_N_PAR + 1)];
	__global double* ytemp = &bufYtemp[commonSize];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqcof_curve2(CUDA_LCC, CUDA_CC, (*CUDA_LCC).covar, (*CUDA_LCC).da, brightness, weight, sig, mDytemp, ytemp, ia, inrel, lpoints);
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2Curve1Last(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global double* bufDytemp,
	__global double* bufYtemp,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	int commonSize = blockIdx.x * ((*CUDA_CC).maxLcPoints + 1);
	__global double* mDytemp = &bufDytemp[commonSize * (MAX_N_PAR + 1)];
	__global double* ytemp = &bufYtemp[commonSize];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	__local double res[BLOCK_DIM];

	mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, mDytemp, ytemp, res, inrel, lpoints);
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqcof2End(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	(*CUDA_LCC).Chisq = mrqcof_end(CUDA_LCC, CUDA_CC, (*CUDA_LCC).covar);

	//if (blockIdx.x == 0)
	//	printf("[%3d] Chisq: %10.7f\n", threadIdx.x, (*CUDA_LCC).Chisq);
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter1Mrqmin2End(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Mrqmin2End\n");

	//mrqmin_2_end(CUDA_LCC, CUDA_ia, CUDA_ma);
	mrqmin_2_end(CUDA_LCC, CUDA_CC);

	(*CUDA_LCC).Niter++;

	//if (blockIdx.x == 0)
	//	printf("[%3d] Niter: %d\n", threadIdx.x, (*CUDA_LCC).Niter);
	//printf("|");
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateIter2(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC)
{
	int i, j;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	//if (blockIdx.x == 0)
	//	printf("[%3d] isNiter: %d\n", threadIdx.x, (*CUDA_LCC).isNiter);

	if ((*CUDA_LCC).isNiter)
	{
		if ((*CUDA_LCC).Niter == 1 || (*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
		{
			if (threadIdx.x == 0)
			{
				(*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;
			}

			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

			int brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
			if ((*CUDA_CC).Numfac % BLOCK_DIM) brtmph++;
			int brtmpl = threadIdx.x * brtmph;
			brtmph = brtmpl + brtmph;
			if (brtmph > (*CUDA_CC).Numfac) brtmph = (*CUDA_CC).Numfac;
			brtmpl++;

			curv(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, brtmpl, brtmph);

			if (threadIdx.x == 0)
			{
				for (i = 1; i <= 3; i++)
				{
					(*CUDA_LCC).chck[i] = 0;


					for (j = 1; j <= (*CUDA_CC).Numfac; j++)
					{
						double qq;
						qq = (*CUDA_LCC).chck[i] + (*CUDA_LCC).Area[j] * (*CUDA_CC).Nor[j][i - 1];

						//if (blockIdx.x == 0)
						//	printf("[%d] [%d][%3d] qq: %10.7f, chck[%d]: %10.7f, Area[%3d]: %10.7f, Nor[%3d][%d]: %10.7f\n",
						//		blockIdx.x, i, j, qq, i, (*CUDA_LCC).chck[i], j, (*CUDA_LCC).Area[j], j, i - 1, (*CUDA_CC).Nor[j][i - 1]);

						(*CUDA_LCC).chck[i] = qq;
					}

					//if (blockIdx.x == 0)
					//	printf("[%d] chck[%d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).chck[i]);
				}

				//printf("[%d] chck[1]: %10.7f, chck[2]: %10.7f, chck[3]: %10.7f\n", blockIdx.x, (*CUDA_LCC).chck[1], (*CUDA_LCC).chck[2], (*CUDA_LCC).chck[3]);

				(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - (pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0)) * pow((*CUDA_CC).conw_r, 2.0);
				//(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1] + (*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2] + (*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]) * ((*CUDA_CC).conw_r * (*CUDA_CC).conw_r);
			}
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // TEST

		if (threadIdx.x == 0)
		{
			//if (blockIdx.x == 0)
			//	printf("ndata - 3: %3d\n", (*CUDA_CC).ndata - 3);

			(*CUDA_LCC).dev_new = sqrt((*CUDA_LCC).rchisq / ((*CUDA_CC).ndata - 3));

			//if (blockIdx.x == 233)
			//{
			//	double dev_best = (*CUDA_LCC).dev_new * (*CUDA_LCC).dev_new * ((*CUDA_CC).ndata - 3);
			//	printf("[%3d] rchisq: %12.8f, ndata-3: %3d, dev_new: %12.8f, dev_best: %12.8f\n",
			//		blockIdx.x, (*CUDA_LCC).rchisq, (*CUDA_CC).ndata - 3, (*CUDA_LCC).dev_new, dev_best);
			//}

			// NOTE: only if this step is better than the previous, 1e-10 is for numeric errors
			if ((*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new > 1e-10)
			{
				(*CUDA_LCC).iter_diff = (*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new;
				(*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
			}
			//		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // TEST
	}
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

__kernel void ClCalculateFinishPole(
	__global struct mfreq_context* CUDA_mCC,
	__global struct freq_context* CUDA_CC,
	__global struct freq_result* CUDA_FR)
{
	int i;
	int3 blockIdx;
	blockIdx.x = get_group_id(0);

	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	//const auto CUDA_LFR = &CUDA_FR[blockIdx.x];
	__global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	double totarea = 0;
	for (i = 1; i <= (*CUDA_CC).Numfac; i++)
	{
		totarea = totarea + (*CUDA_LCC).Area[i];
	}

	//if(blockIdx.x == 2)
	//	printf("[%d] chck[1]: %10.7f, chck[2]: %10.7f, chck[3]: %10.7f, conw_r: %10.7f\n", blockIdx.x, (*CUDA_LCC).chck[1], (*CUDA_LCC).chck[2], (*CUDA_LCC).chck[3], (*CUDA_CC).conw_r);

	//if (blockIdx.x == 2)
	//	printf("rchisq: %10.7f, Chisq: %10.7f \n", (*CUDA_LCC).rchisq, (*CUDA_LCC).Chisq);

	//const double sum = pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0);
	const double sum = ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1]) + ((*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2]) + ((*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]);
	//printf("[FinishPole] [%d] sum: %10.7f\n", blockIdx.x, sum);

	const double dark = sqrt(sum);

	//if (blockIdx.x == 232 || blockIdx.x == 233)
	//	printf("[%d] sum: %12.8f, dark: %12.8f, totarea: %12.8f, dark_best: %12.8f\n", blockIdx.x, sum, dark, totarea, dark / totarea * 100);

	/* period solution */
	const double period = 2 * PI / (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3];

	/* pole solution */
	const double la_tmp = RAD2DEG * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2];

	//if (la_tmp < 0.0)
	//	printf("[CalculateFinishPole] la_best: %4.0f\n", la_tmp);

	const double be_tmp = 90 - RAD2DEG * (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1];

	//if (blockIdx.x == 2)
		//printf("[%d] dev_new: %10.7f, dev_best: %10.7f\n", blockIdx.x, (*CUDA_LCC).dev_new, (*CUDA_LFR).dev_best);

	if ((*CUDA_LCC).dev_new < (*CUDA_LFR).dev_best)
	{
		(*CUDA_LFR).dev_best = (*CUDA_LCC).dev_new;
		(*CUDA_LFR).dev_best_x2 = (*CUDA_LCC).rchisq;
		(*CUDA_LFR).per_best = period;
		(*CUDA_LFR).dark_best = dark / totarea * 100;
		(*CUDA_LFR).la_best = la_tmp < 0 ? la_tmp + 360.0 : la_tmp;
		(*CUDA_LFR).be_best = be_tmp;

		//printf("[%d] dev_best: %12.8f\n", blockIdx.x, (*CUDA_LFR).dev_best);

		//if (blockIdx.x == 232)
		//{
		//	double dev_best = (*CUDA_LFR).dev_best * (*CUDA_LFR).dev_best * ((*CUDA_CC).ndata - 3);
		//	printf("[%3d] rchisq: %12.8f, ndata-3: %3d, dev_new: %12.8f, dev_best: %12.8f\n",
		//		blockIdx.x, (*CUDA_LCC).rchisq, (*CUDA_CC).ndata - 3, (*CUDA_LFR).dev_best, dev_best);
		//}
	}

	if (isnan((*CUDA_LFR).dark_best) == 1)
	{
		(*CUDA_LFR).dark_best = 1.0;
	}

	//if (blockIdx.x == 2)
	//	printf("dark_best: %10.7f \n", (*CUDA_LFR).dark_best);

	//debug
	/*	(*CUDA_LFR).dark=dark;
	(*CUDA_LFR).totarea=totarea;
	(*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
	(*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
	(*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}
