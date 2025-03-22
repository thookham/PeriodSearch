//computes integrated brightness of all visible and iluminated areas
//  and its derivatives

//  8.11.2006


void matrix_neo(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	__global double* bufTim,
	__global double* bufEe,
	__global double* bufEe0,
	__global double* mJpScale,
	__global double* mJpDphp1,
	__global double* mJpDphp2,
	__global double* mJpDphp3,
	__global double* mE1,
	__global double* mE2,
	__global double* mE3,
	__global double* mE01,
	__global double* mE02,
	__global double* mE03,
	__global double* mDe,
	__global double* mDe0,
	int lnp1,
	int Lpoints,
	int num)
{
	__private double f, cf, sf, pom, pom0, alpha;
	__private double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
	__private int lnp;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl, index;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;
	int C = 4;
	int D = 4;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	lnp = lnp1 + brtmpl - 1;
	//printf("lnp: %3d = lnp1: %3d + brtmpl: %3d - 1 | lnp++: %3d\n", lnp, lnp1, brtmpl, lnp + 1);

	int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%3d] cg[%3d]: %10.7f\n", blockIdx.x,  q, (*CUDA_LCC).cg[q]);

	for (int jp = brtmpl; jp <= brtmph; jp++)
	{
		lnp++;

		ee_1 = bufEe[lnp * 3 + 0];		// position vectors
		ee0_1 = bufEe0[lnp * 3 + 0];
		ee_2 = bufEe[lnp * 3 + 1];
		ee0_2 = bufEe0[lnp * 3 + 1];
		ee_3 = bufEe[lnp * 3 + 2];
		ee0_3 = bufEe0[lnp * 3 + 2];

		t = bufTim[lnp];
		alpha = acos(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3);

		/* Exp-lin model (const.term=1.) */
		double f = exp(-alpha / cg[(*CUDA_CC).Ncoef0 + 2]);	//f is temp here
		mJpScale[jp] = 1 + cg[(*CUDA_CC).Ncoef0 + 1] * f + (cg[(*CUDA_CC).Ncoef0 + 3] * alpha);

		//(*CUDA_LCC).jp_dphp_1[jp] = f;
		//(*CUDA_LCC).jp_dphp_2[jp] = cg[(*CUDA_CC).Ncoef0 + 1] * f * alpha / (cg[(*CUDA_CC).Ncoef0 + 2] * cg[(*CUDA_CC).Ncoef0 + 2]);
		//(*CUDA_LCC).jp_dphp_3[jp] = alpha;
		mJpDphp1[jp] = f;
		mJpDphp2[jp] = cg[(*CUDA_CC).Ncoef0 + 1] * f * alpha / (cg[(*CUDA_CC).Ncoef0 + 2] * cg[(*CUDA_CC).Ncoef0 + 2]);
		mJpDphp3[jp] = alpha;

		//  matrix start
		f = cg[(*CUDA_CC).Ncoef0] * t + (*CUDA_CC).Phi_0;
		f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
		sf = sincos(f, &cf);

		/* rotation matrix, Z axis, angle f */
		tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		//(*CUDA_LCC).e_1[jp] = pom + tmat * ee_3;
		mE1[jp] = pom + tmat * ee_3;
		//(*CUDA_LCC).e0_1[jp] = pom0 + tmat * ee0_3;
		mE01[jp] = pom0 + tmat * ee0_3;

		tmat = (-sf) * (*CUDA_LCC).Blmat[1][1] + cf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][2] + cf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][3] + cf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		//(*CUDA_LCC).e_2[jp] = pom + tmat * ee_3;
		mE2[jp] = pom + tmat * ee_3;
		//(*CUDA_LCC).e0_2[jp] = pom0 + tmat * ee0_3;
		mE02[jp] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 1 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 1 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 1 * (*CUDA_LCC).Blmat[3][3];
		//(*CUDA_LCC).e_3[jp] = pom + tmat * ee_3;
		mE3[jp] = pom + tmat * ee_3;
		//(*CUDA_LCC).e0_3[jp] = pom0 + tmat * ee0_3;
		mE03[jp] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[1][1][1] + sf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][2] + sf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][3] + sf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		//(*CUDA_LCC).de[jp][1][1] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][1][1] = pom0 + tmat * ee0_3;
		index = jp * 16 + 1 * 4 + 1;
		mDe[index] = pom + tmat * ee_3;     // <<<<<<<<<<<<<<<<<<<<<<<<<<<<
		mDe0[index] = pom0 + tmat * ee0_3;  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<

		//if (blockIdx.x == 1)
		//{
		//	int mX = (jp * C * D) + (1 * D) + 1;
		//	printf("matrix_neo(): jp[%4d] de[%4d][%d][%d] %9.6f mDe[%6d] %9.6f\n", jp, jp, 1, 1, (*CUDA_LCC).de[jp][1][1], mX, mDe[mX]);
		//}


		tmat = cf * (*CUDA_LCC).Dblm[2][1][1] + sf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][2] + sf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][3] + sf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		//(*CUDA_LCC).de[jp][1][2] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][1][2] = pom0 + tmat * ee0_3;
		index++;
		mDe[index] = pom + tmat * ee_3;
		mDe0[index] = pom0 + tmat * ee0_3;

		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][1] + (t * cf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][2] + (t * cf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][3] + (t * cf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		//(*CUDA_LCC).de[jp][1][3] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][1][3] = pom0 + tmat * ee0_3;
		index++;
		mDe[index] = pom + tmat * ee_3;
		mDe0[index] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[1][1][1] + cf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][2] + cf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][3] + cf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		//(*CUDA_LCC).de[jp][2][1] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][2][1] = pom0 + tmat * ee0_3;
		index = jp * 16 + 2 * 4 + 1;
		mDe[index] = pom + tmat * ee_3;
		mDe0[index] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[2][1][1] + cf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][2] + cf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][3] + cf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		//(*CUDA_LCC).de[jp][2][2] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][2][2] = pom0 + tmat * ee0_3;
		index++;
		mDe[index] = pom + tmat * ee_3;
		mDe0[index] = pom0 + tmat * ee0_3;

		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][1] + (-t * sf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][2] + (-t * sf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][3] + (-t * sf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		//(*CUDA_LCC).de[jp][2][3] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][2][3] = pom0 + tmat * ee0_3;
		index++;
		mDe[index] = pom + tmat * ee_3;
		mDe0[index] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[1][1][1] + 0 * (*CUDA_LCC).Dblm[1][2][1] + 1 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][2] + 0 * (*CUDA_LCC).Dblm[1][2][2] + 1 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][3] + 0 * (*CUDA_LCC).Dblm[1][2][3] + 1 * (*CUDA_LCC).Dblm[1][3][3];
		//(*CUDA_LCC).de[jp][3][1] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][3][1] = pom0 + tmat * ee0_3;
		index = jp * 16 + 3 * 4 + 1;
		mDe[index] = pom + tmat * ee_3;
		mDe0[index] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[2][1][1] + 0 * (*CUDA_LCC).Dblm[2][2][1] + 1 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][2] + 0 * (*CUDA_LCC).Dblm[2][2][2] + 1 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][3] + 0 * (*CUDA_LCC).Dblm[2][2][3] + 1 * (*CUDA_LCC).Dblm[2][3][3];
		//(*CUDA_LCC).de[jp][3][2] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][3][2] = pom0 + tmat * ee0_3;
		index++;
		mDe[index] = pom + tmat * ee_3;
		mDe0[index] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		//(*CUDA_LCC).de[jp][3][3] = pom + tmat * ee_3;
		//(*CUDA_LCC).de0[jp][3][3] = pom0 + tmat * ee0_3;
		index++;
		mDe[index] = pom + tmat * ee_3;
		mDe0[index] = pom0 + tmat * ee0_3;
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);  //__syncthreads();
}

void bright(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global double* cg,
	//__global double* mDe,
	//__global double* mDe0,
	double sJpScale,
	double sJpDphp1,
	double sJpDphp2,
	double sJpDphp3,
	double sE1,
	double sE2,
	double sE3,
	double sE01,
	double sE02,
	double sE03,
	__global double* sDe,
	__global double* sDe0,
	__global double* dytemp,
	__global double* ytemp,
	int jp,
	int Lpoints1,
	int Inrel)
{
	double cl, cls, dnom, s, Scale;
	double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];
	int ncoef0, ncoef, i, j, incl_count = 0;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);
	int x = blockIdx.x;

	ncoef0 = (*CUDA_CC).Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = (*CUDA_CC).ma;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

	/* matrix from neo */
	/* derivatives */
	//e_1 = (*CUDA_LCC).e_1[jp];
	//e_2 = (*CUDA_LCC).e_2[jp];
	//e_3 = (*CUDA_LCC).e_3[jp];
	//e0_1 = (*CUDA_LCC).e0_1[jp];
	//e0_2 = (*CUDA_LCC).e0_2[jp];
	//e0_3 = (*CUDA_LCC).e0_3[jp];
	e_1 = sE1;
	e_2 = sE2;
	e_3 = sE3;
	e0_1 = sE01;
	e0_2 = sE02;
	e0_3 = sE03;

	int B = ((*CUDA_CC).maxLcPoints + 1);
	int C = 4;
	int D = 4;

	//int sX = 1 * D + 1;
	//if ((*CUDA_LCC).de[jp][1][1] != sDe[sX])
	//{
	//	printf("bright(): jp[%4d] de[%4d][%d][%d] %9.6f sDe[%6d] %9.6f false\n", jp, jp, 1, 1, (*CUDA_LCC).de[jp][1][1], sX, sDe[sX]);
	//}
	de[1][1] = sDe[1 * D + 1]; // = (*CUDA_LCC).de[jp][1][1];
	de[1][2] = sDe[1 * D + 2]; // = (*CUDA_LCC).de[jp][1][2];
	de[1][3] = sDe[1 * D + 3]; // = (*CUDA_LCC).de[jp][1][3];
	de[2][1] = sDe[2 * D + 1]; // = (*CUDA_LCC).de[jp][2][1];
	de[2][2] = sDe[2 * D + 2]; // = (*CUDA_LCC).de[jp][2][2];
	de[2][3] = sDe[2 * D + 3]; // = (*CUDA_LCC).de[jp][2][3];
	de[3][1] = sDe[3 * D + 1]; // = (*CUDA_LCC).de[jp][3][1];
	de[3][2] = sDe[3 * D + 2]; // = (*CUDA_LCC).de[jp][3][2];
	de[3][3] = sDe[3 * D + 3]; // = (*CUDA_LCC).de[jp][3][3];
	de0[1][1] = sDe0[1 * D + 1]; // = (*CUDA_LCC).de0[jp][1][1];
	de0[1][2] = sDe0[1 * D + 2]; // = (*CUDA_LCC).de0[jp][1][2];
	de0[1][3] = sDe0[1 * D + 3]; // = (*CUDA_LCC).de0[jp][1][3];
	de0[2][1] = sDe0[2 * D + 1]; // = (*CUDA_LCC).de0[jp][2][1];
	de0[2][2] = sDe0[2 * D + 2]; // = (*CUDA_LCC).de0[jp][2][2];
	de0[2][3] = sDe0[2 * D + 3]; // = (*CUDA_LCC).de0[jp][2][3];
	de0[3][1] = sDe0[3 * D + 1]; // = (*CUDA_LCC).de0[jp][3][1];
	de0[3][2] = sDe0[3 * D + 2]; // = (*CUDA_LCC).de0[jp][3][2];
	de0[3][3] = sDe0[3 * D + 3]; // = (*CUDA_LCC).de0[jp][3][3];

	/*Integrated brightness (phase coeff. used later) */
	double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	short int incl[MAX_N_FAC];
	double dbr[MAX_N_FAC];

	br = 0;
	tmp1 = 0;
	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	tmp5 = 0;

	j = 1;
	for (i = 1; i <= (*CUDA_CC).Numfac; i++, j++)
	{
		lmu = e_1 * (*CUDA_CC).Nor[i][0] + e_2 * (*CUDA_CC).Nor[i][1] + e_3 * (*CUDA_CC).Nor[i][2];
		lmu0 = e0_1 * (*CUDA_CC).Nor[i][0] + e0_2 * (*CUDA_CC).Nor[i][1] + e0_3 * (*CUDA_CC).Nor[i][2];

		if ((lmu > TINY) && (lmu0 > TINY))
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + cls / dnom);
			ar = (*CUDA_LCC).Area[j];
			br += ar * s;

			incl[incl_count] = i;
			dbr[incl_count] = (*CUDA_CC).Darea[i] * s;
			incl_count++;

			double lmu0_dnom = lmu0 / dnom;
			dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
			double lmu_dnom = lmu / dnom;
			dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;


			sum1 = (*CUDA_CC).Nor[i][0] * de[1][1] + (*CUDA_CC).Nor[i][1] * de[2][1] + (*CUDA_CC).Nor[i][2] * de[3][1];
			sum10 = (*CUDA_CC).Nor[i][0] * de0[1][1] + (*CUDA_CC).Nor[i][1] * de0[2][1] + (*CUDA_CC).Nor[i][2] * de0[3][1];
			tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
			sum2 = (*CUDA_CC).Nor[i][0] * de[1][2] + (*CUDA_CC).Nor[i][1] * de[2][2] + (*CUDA_CC).Nor[i][2] * de[3][2];
			sum20 = (*CUDA_CC).Nor[i][0] * de0[1][2] + (*CUDA_CC).Nor[i][1] * de0[2][2] + (*CUDA_CC).Nor[i][2] * de0[3][2];
			tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
			sum3 = (*CUDA_CC).Nor[i][0] * de[1][3] + (*CUDA_CC).Nor[i][1] * de[2][3] + (*CUDA_CC).Nor[i][2] * de[3][3];
			sum30 = (*CUDA_CC).Nor[i][0] * de0[1][3] + (*CUDA_CC).Nor[i][1] * de0[2][3] + (*CUDA_CC).Nor[i][2] * de0[3][3];
			tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

			tmp4 += lmu * lmu0 * ar;
			tmp5 += ar * lmu * lmu0 / (lmu + lmu0);
		}
	}

	//Scale = (*CUDA_LCC).jp_Scale[jp];
	Scale = sJpScale;

	i = jp + (ncoef0 - 3 + 1) * Lpoints1;
	/* Ders. of brightness w.r.t. rotation parameters */
	//(*CUDA_LCC).dytemp[i] = Scale * tmp1;
	dytemp[i] = Scale * tmp1;

	i += Lpoints1;
	//(*CUDA_LCC).dytemp[i] = Scale * tmp2;
	dytemp[i] = Scale * tmp2;
	i += Lpoints1;
	//(*CUDA_LCC).dytemp[i] = Scale * tmp3;
	dytemp[i] = Scale * tmp3;

	i += Lpoints1;
	/* Ders. of br. w.r.t. phase function params. */
	//(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_1[jp];
	//(*CUDA_LCC).dytemp[i] = br * sJpDphp1;
	dytemp[i] = br * sJpDphp1;

	i += Lpoints1;
	//(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_2[jp];
	//(*CUDA_LCC).dytemp[i] = br * sJpDphp2;
	dytemp[i] = br * sJpDphp2;

	i += Lpoints1;
	//(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_3[jp];
	//(*CUDA_LCC).dytemp[i] = br * sJpDphp3;
	dytemp[i] = br * sJpDphp3;

	/* Ders. of br. w.r.t. cl, cls */
	//(*CUDA_LCC).dytemp[jp + (ncoef - 1) * (Lpoints1)] = Scale * tmp4 * cl;
	//(*CUDA_LCC).dytemp[jp + (ncoef) * (Lpoints1)] = Scale * tmp5;
	dytemp[jp + (ncoef - 1) * (Lpoints1)] = Scale * tmp4 * cl;
	dytemp[jp + (ncoef) * (Lpoints1)] = Scale * tmp5;

	/* Scaled brightness */
	//(*CUDA_LCC).ytemp[jp] = br * Scale;
	ytemp[jp] = br * Scale;

	ncoef0 -= 3;
	int m, m1, mr, iStart;
	int d, d1, dr;

	iStart = Inrel + 1;
	m = iStart * (*CUDA_CC).Numfac1;
	d = jp + (Lpoints1 << Inrel);

	m1 = m + (*CUDA_CC).Numfac1;
	mr = 2 * (*CUDA_CC).Numfac1;
	d1 = d + Lpoints1;
	dr = 2 * Lpoints1;

	/* Derivatives of brightness w.r.t. g-coeffs */
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += 2, m += mr, m1 += mr, d += dr, d1 += dr)
		{
			double tmp = 0, tmp1 = 0;
			double l_dbr = dbr[0];
			int l_incl = incl[0];
			tmp = l_dbr * (*CUDA_LCC).Dg[m + l_incl];
			if ((i + 1) <= ncoef0)
			{
				tmp1 = l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
			}

			for (j = 1; j < incl_count; j++)
			{
				double l_dbr = dbr[j];
				int l_incl = incl[j];
				tmp += l_dbr * (*CUDA_LCC).Dg[m + l_incl];
				if ((i + 1) <= ncoef0)
				{
					tmp1 += l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
				}
			}

			//(*CUDA_LCC).dytemp[d] = Scale * tmp;
			dytemp[d] = Scale * tmp;
			if ((i + 1) <= ncoef0)
			{
				//(*CUDA_LCC).dytemp[d1] = Scale * tmp1;
				dytemp[d1] = Scale * tmp1;
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d += Lpoints1)
		{
			//(*CUDA_LCC).dytemp[d] = 0;
			dytemp[d] = 0;
		}
	}

	//return(0);
}
