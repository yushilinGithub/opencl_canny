
#define PIX_PER_WI 8
#define loadpix(addr) *(__global int *)(addr)
#define storepix(val, addr) *(__global int *)(addr) = (int)(val)
#define LOCAL_TOTAL (GRP_SIZEX*GRP_SIZEY)
#define l_stack_size (4*LOCAL_TOTAL)
#define p_stack_size 8
#define positive 255



__constant short move_dir[2][8] = {
    { -1, -1, -1, 0, 0, 1, 1, 1 },
    { -1, 0, 1, -1, 1, -1, 0, 1 }
};

__kernel void stage2_hysteresis(__global uchar *map_ptr, int rows, int cols)
{


    int x = get_global_id(1);//width
    int y = get_global_id(0) * PIX_PER_WI;//height

    int lid = get_local_id(1) + get_local_id(0) * GRP_SIZEX; //x+ y*GRP_SIZEX

    __local ushort2 l_stack[l_stack_size];
    __local int l_counter;

    if (lid == 0)
        l_counter = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < cols)
    {
        //__global uchar* map = map_ptr + mad24(y, map_step, x * (int)sizeof(int));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI; ++cy)
        {
            if (y < rows)
            {
                int type = map_ptr[mad24(y, cols, x)];
                if (type == 2)
                {
                    l_stack[atomic_inc(&l_counter)] = (ushort2)(x, y);
                    map_ptr[mad24(y, cols, x)] = positive;
                }
                y++;

            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ushort2 p_stack[p_stack_size];
    int p_counter = 0;

    while(l_counter != 0)
    {
        int mod = l_counter % LOCAL_TOTAL;
        int pix_per_thr = l_counter / LOCAL_TOTAL + ((lid < mod) ? 1 : 0);

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < pix_per_thr; ++i)
        {
            int index = atomic_dec(&l_counter) - 1;
            if (index < 0)
               continue;
            ushort2 pos = l_stack[ index ];

            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                ushort posx = pos.x + move_dir[0][j];
                ushort posy = pos.y + move_dir[1][j];
                if (posx < 0 || posy < 0 || posx >= cols || posy >= rows)
                    continue;

                int type = map_ptr[mad24(posy, cols, posx)];
                if (type == 0)
                {
                    p_stack[p_counter++] = (ushort2)(posx, posy);
                    if(posy<rows-1) map_ptr[mad24(posy, cols, posx)] = positive;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_counter < 0)
            l_counter = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        while (p_counter > 0)
        {
            l_stack[ atomic_inc(&l_counter) ] = p_stack[--p_counter];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


// Get the edge result. edge type of value 2 will be marked as an edge point and set to 255. Otherwise 0.
// map      edge type mappings
// dst      edge output

__kernel void getEdges(__global const uchar *mapptr, int map_step, int map_offset, int rows, int cols,
                       __global uchar *dst, int dst_step, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI;

    if (x < cols)
    {
        int map_index = mad24(map_step, y, mad24(x, (int)sizeof(int), map_offset));
        int dst_index = mad24(dst_step, y, x + dst_offset);

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI; ++cy)
        {
            if (y < rows)
            {
                __global const int * map = (__global const int *)(mapptr + map_index);
                dst[dst_index] = (uchar)(-(map[0] >> 1));

                y++;
                map_index += map_step;
                dst_index += dst_step;
            }
        }
    }
}


