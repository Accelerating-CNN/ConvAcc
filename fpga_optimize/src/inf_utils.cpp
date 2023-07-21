#include "inf_utils.h"



int classImage(Tensor * Z)
{
	int maxel = std::distance((*Z)[0][0],std::max_element((*Z)[0][0],(*Z)[0][0] + 100));
	printf("Predicted class with %f : %s \n",(*Z)[0][0][maxel],classes_img100[maxel]);
	return maxel;
}

uint8_t read8(FILE * f)
{
	uint8_t val;
	if( 1 != fread(&val,1,1,f)){
		printf("ERROR reading Little Endian!\n");
		return 0xFF;
	}
	return val;
}

uint16_t read16(FILE *f)
{
	uint16_t val = read8(f) | (read8(f) << 8);
	return val;
}

uint32_t read32(FILE * f)
{
	uint32_t val = read16(f) | (read16(f) << 16 );
	return val;
}


Tensor * readBMP(const char * infile)
{
	FILE * f = fopen(infile,"rb");
	uint16_t val = read16(f);
	if(val != 0x4d42){
		printf("ERROR wrong bmp val!\n");
		return NULL;
	}
	uint32_t size = read32(f);
#ifdef BMP_DEBUG
	printf("BMP size: %u \n",size);
#endif
	/* Discard 4 bytes */
	read32(f);
	uint32_t offset = read32(f);
#ifdef BMP_DEBUG
	printf("BMP offset: %u \n",offset);
#endif
	uint32_t header_size = read32(f);
	uint32_t width = read32(f);
	uint32_t height = read32(f);
#ifdef BMP_DEBUG
	printf("Img: %ux%u \n",width,height);
#endif
	read16(f);
	uint16_t bpp = read16(f);
	if(bpp != 24){
		printf("BYTE ERROR!");
	}
	if((width != 128) && (height != 128)){
		printf("ERROR width or height has to be 128px!\n");
		return NULL;
	}
	if(fseek(f,offset,SEEK_SET)){
		printf("ERROR fseek failed!\n");
		return NULL;
	}
	Tensor * X = new Tensor(3,128,128);
	for(int i = 127 - (128 - height)/2; i > ((128 - height)/2 + height%2); 
			i--){
		for(int j= (128 - width)/2; j< 128 - ((128 - width)/2 + width%2); j++){
			for(int k =2 ; k >= 0; k--){
				(*X)[k][i][j] = (float) read8(f)/255;
			}
		}
		int tot = width * 3;
		while(tot  % 4){
			read8(f);
			tot++;
		}
	}
	fclose(f);
	return  X ;
}
