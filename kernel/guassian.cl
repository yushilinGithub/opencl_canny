#define GROUP_SIZE  GRP_SIZEY*GRP_SIZEX

__constant float gua[3][3] = {{0.0625, 0.125, 0.0625},
                            {0.1250, 0.250, 0.1250},
                            {0.0625, 0.125, 0.0625}};

__kernel void guassian(__global uchar *data,\
                        __global uchar *output,\
                         int rows,\
                         int cols){

                            
             int g_row = get_global_id(0);
             int g_col = get_global_id(1);
             int l_row = get_local_id(0);
             int l_col = get_local_id(1);
             int pos = mad24(g_row,cols,g_col);

            //copy global data to local
            __local int l_data[(GRP_SIZEX+2)*(GRP_SIZEY+2)];

            int lidx = mad24(l_row+1,GRP_SIZEX+2,l_col+1);
            l_data[lidx] = data[pos];

            if(l_row==0){//top column
                int upper_row = max(g_row-1,0);
                l_data[l_col+1] = data[mad24(upper_row,cols,g_col)];
                if(l_col==0) {
                    l_data[0] = data[mad24(upper_row,cols,max(g_col-1,0))];
                    } //top left
                else if(l_col==GRP_SIZEX-1) {
                    l_data[GRP_SIZEX+1] = data[mad24(upper_row,cols,min(g_col+1,cols-1))];
                    }
            }
            else if(l_row==GRP_SIZEY-1){//buttom column
                int buttom_row = min(g_row+1,cols-1);
                l_data[mad24(GRP_SIZEY+1,GRP_SIZEX+2,l_col+1)] = data[pos-cols];
                if(l_col==0) {
                    l_data[(GRP_SIZEY+1)*(GRP_SIZEX+2)] = data[max(pos-cols-1,0)];
                }
                else if(l_col==GRP_SIZEX-1) {
                    l_data[mad24(GRP_SIZEY+1,GRP_SIZEX+2,GRP_SIZEX+1)] = data[pos-cols+1];
                    } 
            }

            if(l_col==0){ //left column
                l_data[(l_row+1)*(GRP_SIZEX+2)] = data[max(pos-1,0)];
            }
            else if(l_col==GRP_SIZEX-1){//right column
                l_data[mad24(l_row+1,GRP_SIZEX+2,GRP_SIZEX+1)] = data[pos+1];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            int sum=0;
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    sum+=gua[i][j]*l_data[mad24(l_row+i,GRP_SIZEX+2,l_col+j)];      
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            sum = min(255,max(0,sum));
            output[pos] =sum;
            
    
    }
