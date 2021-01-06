
#define LDSIZEX GRP_SIZEX+4
#define LDSIZEY GRP_SIZEY+4
#define TDSIZE LDSIZEX*LDSIZEY
#define neg 1
#define positive 2
#define unsure 0
#define loadpix(addr) *(__global const uchar *)(addr)
__constant int sobx[3][3] = { {-1, 0, 1},
                              {-2, 0, 2},
                              {-1, 0, 1} };

__constant int soby[3][3] = { {-1,-2,-1},
                              { 0, 0, 0},
                              { 1, 2, 1} };
__constant float PI = 3.141592654;



inline int kernelcom(__local uchar *l_data,int2 lid){
    int l_row = lid.s0;
    int l_col = lid.s1;
    int m;
    float sum_y=0,sum_x=0;
    for(int i = 0;i<3;i++){
        for(int j = 0;j<3;j++){
            sum_x += sobx[i][j]*l_data[mad24(l_row+i+1,LDSIZEX,l_col+j+1)];
            sum_y += soby[i][j]*l_data[mad24(l_row+i+1,LDSIZEX,l_col+j+1)];
        }
    }
    int mag = hypot(sum_x,sum_y);
    m = min(mag,255);
    return m;
}

inline int2 kernelcomde(__local uchar *l_data,int2 lid){
    int2 m;
    int l_row = lid.s0;
    int l_col = lid.s1;
    float sum_y=0,sum_x=0;
    for(int i = 0;i<3;i++){
        for(int j = 0;j<3;j++){
            sum_x += sobx[i][j]*l_data[mad24(l_row+i+1,LDSIZEX,l_col+j+1)];
            sum_y += soby[i][j]*l_data[mad24(l_row+i+1,LDSIZEX,l_col+j+1)];
        }
    }
    int mag = hypot(sum_x,sum_y);
    m.s0 = min(mag,255);

    float angle=atan2(sum_y,sum_x);
    if(angle<0){
        angle = fmod((angle+2*PI),2*PI);
    }
    m.s1 = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
    return m;
}



__kernel void sobal(__global uchar *data,__global uchar *output,int rows,int cols,int h_threshold,int l_threshold){
    int g_row = get_global_id(0);
    int g_col = get_global_id(1);
    int l_row = get_local_id(0);
    int l_col = get_local_id(1);


    if(g_row >= rows | g_col >= cols) return;

    int pos = mad24(g_row,cols,g_col);
    int maxpos = cols+rows-1;
    __local uchar l_data[LDSIZEX*LDSIZEY];
    __local uchar mag[(GRP_SIZEX+2)*(GRP_SIZEY+2)];

    int mg_pos = mad24(l_row+1,GRP_SIZEX+2,l_col+1); 
    int ld_pos = mad24(l_row+2,LDSIZEX,l_col+2);
    
    l_data[ld_pos] = data[pos];

    if(l_row<=1){//top two column
        l_data[mad24(l_row,LDSIZEX,l_col+2)] = data[pos-2*cols];
        if(l_col<=1){
            l_data[mad24(l_row,LDSIZEX,l_col)] = data[pos-2*cols-2];
        }
        else if(l_col>=GRP_SIZEX-2){
            l_data[mad24(l_row,LDSIZEX,l_col+4)] = data[pos-2*cols+2];
        }
    }
    else if(l_row>=GRP_SIZEY-2){

        l_data[mad24(l_row+4,LDSIZEX,l_col+2)] = data[pos+2*cols];
        if(l_col<=1){
            l_data[mad24(l_row+4,LDSIZEX,l_col)] = data[pos+2*cols-2];
        }
        else if(l_col>=GRP_SIZEX-2){
            l_data[mad24(l_row+4,LDSIZEX,l_col+4)] = data[pos+2*cols+2];
        }
    }

    if(l_col<=1){
        l_data[mad24(l_row+2,LDSIZEX,l_col)] = data[pos-2];
    }else if(l_col>=GRP_SIZEX-2){
        l_data[mad24(l_row+2,LDSIZEX,l_col+4)] = data[pos+2];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    
    int2 index;
    index.s0 = l_row;
    index.s1 = l_col;
    
    int2 mg;
    mg = kernelcomde(l_data,index);
    barrier(CLK_LOCAL_MEM_FENCE);
    mag[mg_pos] = mg.s0;
    int mymag = mg.s0;
    
    int degree = mg.s1;

    int m = 0;
    if(l_row==0){//upper
        index.s0 = l_row-1;
        index.s1 = l_col;
        m = kernelcom(l_data,index);
        mag[l_col+1] = m;

        if(l_col==0){ //top left
            index.s0 = l_row-1;
            index.s1 = l_col-1;
            m = kernelcom(l_data,index);
            mag[0] = m;
        }else if(l_col==GRP_SIZEX-1){//top right
            index.s0 = l_row-1;
            index.s1 = l_col+1;
            m = kernelcom(l_data,index);
            mag[GRP_SIZEX+1] = m;
        }
    }else if(l_row==GRP_SIZEY-1){//buttom
        index.s0 = l_row+1;
        index.s1 = l_col;
        m = kernelcom(l_data,index);
        mag[mad24(l_row+2,GRP_SIZEX+2,l_col+1)] = m;
        if(l_col==0){
            index.s0 = l_row+1;
            index.s1 = l_col-1;
            m = kernelcom(l_data,index);
            mag[(l_row+2)*(GRP_SIZEX+2)] = m;
        }else if(l_col==GRP_SIZEX-1){
            index.s0 = l_row+1;
            index.s1 = l_col+1;
            m = kernelcom(l_data,index);
            mag[mad24(l_row+2,GRP_SIZEX+2,l_col+2)] = m;
        }
    }

    if(l_col==0){
        index.s0 = l_row;
        index.s1 = l_col-1;
        m=  kernelcom(l_data,index);
        mag[(l_row+1)*(GRP_SIZEX+2)]=m;
    }else if(l_col==GRP_SIZEX-1){
        index.s0 = l_row;
        index.s1 = l_col+1;
        m = kernelcom(l_data,index);
        mag[mad24(l_row+1,GRP_SIZEX+2,l_col+2)]=m;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    l_row++;
    l_col++;
    //non max suppresion
    if(mymag<l_threshold){
        output[pos] = neg;
    }
    else{
        switch(degree){
            // A gradient angle of 0 degrees = an edge that is North/South
            // Check neighbors to the East and West
            case 0:
                // supress me if my neighbor has larger magnitude
                if (mymag <= mag[mad24(l_row,GRP_SIZEX+2,l_col+1)] || mymag <= mag[mad24(l_row,GRP_SIZEX+2,l_col-1)]){   // west
                    output[pos] = neg;
                }else{
                    output[pos] = (mymag>h_threshold)?positive:unsure;
                }break;
            // A gradient angle of 45 degrees = an edge that is NW/SE
            // Check neighbors to the NE and SW
            case 45:
                // supress me if my neighbor has larger magnitude
                if (mymag <= mag[mad24(l_row-1,GRP_SIZEX+2,l_col+1)] || mymag <= mag[mad24(l_row+1,GRP_SIZEX+2,l_col-1)]){   // south west
                    output[pos] = neg;
                }else{
                    output[pos] = (mymag>h_threshold)?positive:unsure;
                }break;
                        
            // A gradient angle of 90 degrees = an edge that is E/W
            // Check neighbors to the North and South
            case 90:
                // supress me if my neighbor has larger magnitude
                if (mymag <= mag[mad24(l_row-1,GRP_SIZEX+2,l_col)] || mymag <= mag[mad24(l_row+1,GRP_SIZEX+2,l_col)]){   // south
                    output[pos] = neg;
                }else{
                    output[pos] = (mymag>h_threshold)?positive:unsure;
                }break;  
            // A gradient angle of 135 degrees = an edge that is NE/SW
            // Check neighbors to the NW and SE
            case 135:
                // supress me if my neighbor has larger magnitude
                if (mymag <= mag[mad24(l_row-1,GRP_SIZEX+2,l_col-1)] || mymag <= mag[mad24(l_row+1,GRP_SIZEX+2,l_col+1)]){   // south east
                    output[pos] = neg;
                }else{
                    output[pos] = (mymag>h_threshold)?positive:unsure;
                }break;
        } 
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
}