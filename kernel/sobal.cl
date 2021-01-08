#define TG22 0.4142135623730950488016887242097f
#define TG67 2.4142135623730950488016887242097f


#if cn == 1
#define loadpix(addr) convert_floatN(*(__global const TYPE *)(addr))
#else
#define loadpix(addr) convert_floatN(vload3(0, (__global const TYPE *)(addr)))
#endif
#define storepix(value, addr) *(__global int *)(addr) = (int)(value)

__constant int neg = 1,positive=2,unsure = 0;
/*
    stage1_with_sobel:
        Sobel operator
        Calc magnitudes
        Non maxima suppression
        Double thresholding
*/

__constant int prev[4][2] = {
    { 0, -1 },
    { -1, 1 },
    { -1, 0 },
    { -1, -1 }
};

__constant int next[4][2] = {
    { 0, 1 },
    { 1, -1 },
    { 1, 0 },
    { 1, 1 }
};

inline float3 sobel(int idx, __local const float *smem)
{
    // result: x, y, mag
    float3 res;

    float dx = fma((float)2, smem[idx + GRP_SIZEX + 6] - smem[idx + GRP_SIZEX + 4],
        smem[idx + 2] - smem[idx] + smem[idx + 2 * GRP_SIZEX + 10] - smem[idx + 2 * GRP_SIZEX + 8]);

    float dy = fma((float)2, smem[idx + 1] - smem[idx + 2 * GRP_SIZEX + 9],
        smem[idx + 2] - smem[idx + 2 * GRP_SIZEX + 10] + smem[idx] - smem[idx + 2 * GRP_SIZEX + 8]);


    float magN = fma(dx, dx, dy * dy);
    //float magN = fabs(dx)+ fabs(dy);

    res.z = magN;
    res.x = dx;
    res.y = dy;

    return res;
}

__kernel void stage1_with_sobel(__global uchar const *src,  __global uchar *map,\
                                int rows, int cols, float high_thr, float low_thr)
{
    __local float smem[(GRP_SIZEX + 4) * (GRP_SIZEY + 4)];

    int lidx = get_local_id(1);
    int lidy = get_local_id(0);

    int start_x = GRP_SIZEX * get_group_id(1);
    int start_y = GRP_SIZEY * get_group_id(0);

    int i = lidx + lidy * GRP_SIZEX;
    for (int j = i;  j < (GRP_SIZEX + 4) * (GRP_SIZEY + 4); j += GRP_SIZEX * GRP_SIZEY)
    {
        int x = clamp(start_x - 2 + (j % (GRP_SIZEX + 4)), 0, cols - 1); //column
        int y = clamp(start_y - 2 + (j / (GRP_SIZEX + 4)), 0, rows - 1); //row
        smem[j] = src[mad24(y,cols,x)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //// Sobel, Magnitude
    //

    __local float mag[(GRP_SIZEX + 2) * (GRP_SIZEY + 2)];

    lidx++;
    lidy++;

    if (i < GRP_SIZEX + 2) //first two columns
    {
        int grp_sizey = min(GRP_SIZEY + 1, rows - start_y);
        mag[i] = (sobel(i, smem)).z;   //first row
        mag[i + grp_sizey * (GRP_SIZEX + 2)] = (sobel(i + grp_sizey * (GRP_SIZEX + 4), smem)).z; //last row
    }
    if (i < GRP_SIZEY + 2) //first tow rows
    {
        int grp_sizex = min(GRP_SIZEX + 1, cols - start_x);
        mag[i * (GRP_SIZEX + 2)] = (sobel(i * (GRP_SIZEX + 4), smem)).z; //first column
        mag[i * (GRP_SIZEX + 2) + grp_sizex] = (sobel(i * (GRP_SIZEX + 4) + grp_sizex, smem)).z; //second column
    }

    int idx = lidx + lidy * (GRP_SIZEX + 4);
    i = lidx + lidy * (GRP_SIZEX + 2);

    float3 res = sobel(idx, smem);
    mag[i] = res.z;
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = (int) res.x;
    int y = (int) res.y;

    //// Threshold + Non maxima suppression
    //

    /*
        Sector numbers

        3   2   1
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        1   2   3

        We need to determine arctg(dy / dx) to one of the four directions: 0, 45, 90 or 135 degrees.
        Therefore if abs(dy / dx) belongs to the interval
        [0, tg(22.5)]           -> 0 direction
        [tg(22.5), tg(67.5)]    -> 1 or 3
        [tg(67,5), +oo)         -> 2

        Since tg(67.5) = 1 / tg(22.5), if we take
        a = abs(dy / dx) * tg(22.5) and b = abs(dy / dx) * tg(67.5)
        we can get another intervals

        in case a:
        [0, tg(22.5)^2]     -> 0
        [tg(22.5)^2, 1]     -> 1, 3
        [1, +oo)            -> 2

        in case b:
        [0, 1]              -> 0
        [1, tg(67.5)^2]     -> 1,3
        [tg(67.5)^2, +oo)   -> 2

        that can help to find direction without conditions.

        0 - might belong to an edge
        1 - pixel doesn't belong to an edge
        2 - belong to an edge
    */

    int gidx = get_global_id(1);
    int gidy = get_global_id(0);

    if (gidx >= cols || gidy >= rows)
        return;

    float mag0 = mag[i];

    int value = neg;
    if (mag0 > low_thr)
    {
        float x_ = abs(x);
        float y_ = abs(y);

        int a = (y_ * TG22 >= x_) ? 2 : 1;
        int b = (y_ * TG67 >= x_) ? 1 : 0;

        //  a = { 1, 2 }
        //  b = { 0, 1 }
        //  a * b = { 0, 1, 2 } - directions that we need ( + 3 if x ^ y < 0)

        int dir3 = (a * b) & (((x ^ y) & 0x80000000) >> 31); // if a = 1, b = 1, dy ^ dx < 0
        int dir = a * b + 2 * dir3;
        float prev_mag = mag[(lidy + prev[dir][0]) * (GRP_SIZEX + 2) + lidx + prev[dir][1]];
        float next_mag = mag[(lidy + next[dir][0]) * (GRP_SIZEX + 2) + lidx + next[dir][1]] + (dir & 1);

        if (mag0 > prev_mag && mag0 >= next_mag)
        {
            value = (mag0 > high_thr) ? positive : unsure;
        }
    }

    map[mad24(gidy, cols, gidx)] = value;
}