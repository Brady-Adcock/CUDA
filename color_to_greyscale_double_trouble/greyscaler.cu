#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <jpeglib.h>
#include <jerror.h>
#include "wrappers.h"
#include "h_colorToGreyscale.h"
#include "d_colorToGreyscale.h"

#define CHANNELS 3

//prototypes for functions in this file 
void parseCommandArgs(int, char **, int *, int *, char **);
void printUsage();
void readJPGImage(char *, unsigned char **, int *, int *);
void writeJPGImage(char *, unsigned char *, int, int);
char * buildFilename(char *, const char *);
void compare(unsigned char * d_Pout, unsigned char * h_Pout, int size);

/*
    main 
    Opens the jpg file and reads the contents.  Uses the CPU
    and the GPU to perform the greyscale.  Compares the CPU and GPU
    results.  Writes the results to output files.  Outputs the
    time of each.
*/
int main(int argc, char * argv[])
{
    unsigned char * Pin;
    char * fileName;
    int width, height, blkWidth, blkHeight;
    parseCommandArgs(argc, argv, &blkWidth, &blkHeight, &fileName);
    readJPGImage(fileName, &Pin, &width, &height);

    //use the CPU to perform the greyscale
    unsigned char * h_Pout; 
    h_Pout = (unsigned char *) Malloc(sizeof(unsigned char) * width * height);
    float cpuTime = h_colorToGreyscale(h_Pout, Pin, width, height);
    char * h_outfile = buildFilename(fileName, "h_grey");
    writeJPGImage(h_outfile, h_Pout, width, height);

    //use the GPU to perform the greyscale 
    unsigned char * d_Pout; 
    d_Pout = (unsigned char *) Malloc((sizeof(unsigned char) * width * height));
    float gpuTime = d_colorToGreyscale(d_Pout, Pin, width, height, blkWidth, blkHeight);
    char * d_outfile = buildFilename(fileName, "d_grey");
    writeJPGImage(d_outfile, d_Pout, width, height);

    //compare the CPU and GPU results
    compare(d_Pout, h_Pout, width * height);

    printf("CPU time: %f msec\n", cpuTime);
    printf("GPU time: %f msec\n", gpuTime);
    printf("Speedup: %f\n", cpuTime/gpuTime);
    return EXIT_SUCCESS;
}

/* 
    compare
    This function takes two arrays of greyscale pixel values.  One array
    contains pixel values calculated  by the GPU.  The other array contains
    greyscale pixel values calculated by the CPU.  This function checks to
    see that the values are the same within a slight margin of error.

    d_Pout - pixel values calculated by GPU
    h_Pout - pixel values calculated by CPU
    size - size in elements of both arrays
    
    Outputs an error message and exits program if the arrays differ.
*/
void compare(unsigned char * d_Pout, unsigned char * h_Pout, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        //GPU and CPU have different floating point standards so
        //the results could be slightly different
        int diff = d_Pout[i] - h_Pout[i];
        if (abs(diff) > 1)
        {
            printf("Greyscale results don't match.\n");
            printf("CPU pixel %d: %d\n", i, h_Pout[i]);
            printf("GPU pixel %d: %d\n", i, d_Pout[i]);
            exit(EXIT_FAILURE);
        }
    }
}

/* 
    writeJPGImage
    Writes a greyscale jpg image to an output file.

    outfile - name of jpg file (ends with a .jpg extension)
    Pout - array of pixels
    width - width (x-dimension) of image
    height - height (y-dimension) of image
*/
void writeJPGImage(char * filename, unsigned char * Pout, 
                   int width, int height)
{
   struct jpeg_compress_struct cinfo;
   struct jpeg_error_mgr jerr;
   JSAMPROW rowPointer[1];

   //set up error handling
   cinfo.err = jpeg_std_error(&jerr);
   //initialize the compression object
   jpeg_create_compress(&cinfo);

   //open the output file
   FILE * fp;
   if ((fp = fopen(filename, "wb")) == NULL)
   {
     fprintf(stderr, "Can't open %s\n", filename);
     exit(1);
   }
   //initalize state for output to outfile
   jpeg_stdio_dest(&cinfo, fp);

   cinfo.image_width = width;    /* image width and height, in pixels */
   cinfo.image_height = height;
   cinfo.input_components = 1;   /* # of color components per pixel */
   cinfo.in_color_space = JCS_GRAYSCALE;
   jpeg_set_defaults(&cinfo);
   jpeg_set_quality(&cinfo, 75, TRUE);

   //TRUE means it will write a complete interchange-JPEG file
   jpeg_start_compress(&cinfo, TRUE);

   while (cinfo.next_scanline < cinfo.image_height)
   {
      rowPointer[0] = &Pout[cinfo.next_scanline * width];
      (void) jpeg_write_scanlines(&cinfo, rowPointer, 1);
   }
   jpeg_finish_compress(&cinfo);
   fclose(fp);
   jpeg_destroy_compress(&cinfo);
}

/*
    buildFilename
    This function returns the concatenation of two strings by
    first allocating enough space to hold both strings and then
    copying the two strings into the allocated space.  
    It is used by the program to build the output file names.
*/    
char * buildFilename(char * infile, const char * prefix)
{
   int infilelen = strlen(infile);
   int prefixlen = strlen(prefix);
   int len = infilelen + prefixlen + 1; //include space for NULL
   char * outfile = (char *) Malloc(sizeof(char *) * len);

   strncpy(outfile, prefix, prefixlen + 1);
   strncpy(&outfile[strlen(prefix)], infile, infilelen + 1);

   return outfile;
}
   
/*
    readJPGImage
    This function opens a jpg file and reads the contents.  
    Each pixel consists of bytes for red, green, and blue.  
    The array Pin is initialized to the pixel bytes.  width and height
    are pointers to ints that are set to those values.
    filename - name of the .jpg file
*/
void readJPGImage(char * filename, unsigned char ** Pin, 
                  int * width, int * height)
{
   unsigned long dataSize;             // length of the file
   int channels;                       //  3 =>RGB   4 =>RGBA 
   unsigned char * rowptr[1];          // pointer to an array
   unsigned char * jdata;              // data for the image
   struct jpeg_decompress_struct info; //for our jpeg info
   struct jpeg_error_mgr err;          //the error handler

   FILE * fp = fopen(filename, "rb"); //read binary
   if (fp == NULL)
   {
      fprintf(stderr, "Error reading file %s\n", filename);
      printUsage();
   }

   info.err = jpeg_std_error(& err);
   jpeg_create_decompress(&info);

   jpeg_stdio_src(&info, fp);
   jpeg_read_header(&info, TRUE);   // read jpeg file header
   jpeg_start_decompress(&info);    // decompress the file

   //set width and height
   (*width) = info.output_width;
   (*height) = info.output_height;
   channels = info.num_components;
   if (channels != CHANNELS)
   {
      fprintf(stderr, "%s is not an RGB jpeg image\n", filename);
      printUsage();
   }

   dataSize = (*width) * (*height) * channels;
   jdata = (unsigned char *)Malloc(dataSize);
   while (info.output_scanline < info.output_height) // loop
   {
      // Enable jpeg_read_scanlines() to fill our jdata array
      rowptr[0] = (unsigned char *)jdata +  // secret to method
                  channels * info.output_width * info.output_scanline;

      jpeg_read_scanlines(&info, rowptr, 1);
   }
   jpeg_finish_decompress(&info);   //finish decompressing
   jpeg_destroy_decompress(&info);
   fclose(fp);                      //close the file
   (*Pin) = jdata;
   return;
}

/*
    parseCommandArgs
    This function parses the command line arguments. The program can be executed in
    one of two ways:
    ./greyscalar <file>.jpg
    or
    ./greyscalar -w <blkWidth> -h <blkHeight> <file>.jpg
    This function parses the command line arguments, setting block width and block
    height to the command line argument values or to 16 if no command line arguments
    are provided.  In addition, it checks to see if the last command line argument
    is a jpg file and sets (*fileNm) to argv[i] where argv[i] is the name of the jpg
    file.  
*/
void parseCommandArgs(int argc, char * argv[], int * blkWidth, int * blkHeight, char ** fileNm)
{
    int fileIdx = 1, blkW = 16, blkH = 16;
    struct stat buffer;
    if (argc != 2 && argc != 6) printUsage();
    if (argc == 6) 
    {
        fileIdx = 5;
        if (strncmp("-bw", argv[1], 2) != 0) printUsage();
        if (strncmp("-bh", argv[3], 2) != 0) printUsage();
        blkW = atoi(argv[2]);
        blkH = atoi(argv[4]);
        if (blkW <= 0 || blkH <= 0) printUsage();
    }

    int len = strlen(argv[fileIdx]);
    if (len < 5) printUsage();
    if (strncmp(".jpg", &argv[fileIdx][len - 4], 4) != 0) printUsage();

    //stat function returns 1 if file does not exist
    if (stat(argv[fileIdx], &buffer)) printUsage();
    (*blkWidth) = blkW;
    (*blkHeight) = blkH;
    (*fileNm) = argv[fileIdx];
}

/*
    printUsage
    This function is called if there is an error in the command line
    arguments or if the .jpg file that is provided by the command line
    argument is improperly formatted.  It prints usage information and
    exits.
*/
void printUsage()
{
    printf("This application takes as input the name of a .jpg\n");
    printf("file containing a color image and creates a file\n");
    printf("containing a greyscale version of the file.\n");
    printf("\nusage: greyscaler [-bw <blkWidth> -bh <blkHeight>] <name>.jpg\n");
    printf("         <blkWidth> is the width of the blocks created for GPU\n");
    printf("         <blkHeight> is the height of the blocks created for GPU\n");
    printf("         If the -bw and -bh arguments are omitted, the block size\n");
    printf("         defaults to 16 by 16.\n");
    printf("         <name>.jpg is the name of the input jpg file\n");
    printf("Examples:\n");
    printf("./greyscaler color1200by800.jpg\n");
    printf("./greyscaler -bw 8 -bh 16 color1200by800.jpg\n");
    exit(EXIT_FAILURE);
}
