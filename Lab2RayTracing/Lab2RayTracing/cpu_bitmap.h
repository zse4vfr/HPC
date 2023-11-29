/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include "gl_helper.h"
#include <string>
#include <fstream>


struct CPUBitmap {
    unsigned char    *pixels;
    int     x, y;
    void    *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap( int width, int height, void *d = NULL ) {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }

    ~CPUBitmap() {
        delete [] pixels;
    }

    unsigned char* get_ptr( void ) const   { return pixels; }
    long image_size( void ) const { return x * y * 4; }    

    void SaveBMP(char* filename, int wid, int hei, unsigned char* bitmapData)
    {
        int width = wid;
        int height = hei;

        BITMAPFILEHEADER bitmapFileHeader;
        memset(&bitmapFileHeader, 0, sizeof(BITMAPFILEHEADER));
        bitmapFileHeader.bfSize = sizeof(BITMAPFILEHEADER);
        bitmapFileHeader.bfType = 0x4d42;	//BM
        bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

        // заполнить bitmapinfoheader.
        BITMAPINFOHEADER bitmapInfoHeader;
        memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
        bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmapInfoHeader.biWidth = width;
        bitmapInfoHeader.biHeight = height;
        bitmapInfoHeader.biPlanes = 1;
        bitmapInfoHeader.biBitCount = 32;
        bitmapInfoHeader.biCompression = BI_RGB;
        bitmapInfoHeader.biSizeImage = width * abs(height) * 4;

        //////////////////////////////////////////////////////////////////////////
        FILE* filePtr;
        int imageIdx;
        unsigned char tempRGB;

        //swap R B
        for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 4)
        {
            tempRGB = bitmapData[imageIdx];
            bitmapData[imageIdx] = bitmapData[imageIdx + 2];
            bitmapData[imageIdx + 2] = tempRGB;
        }

        filePtr = fopen(filename, "wb");

        fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

        fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

        fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

        fclose(filePtr);

        //swap R B again
        for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 4)
        {
            tempRGB = bitmapData[imageIdx];
            bitmapData[imageIdx] = bitmapData[imageIdx + 2];
            bitmapData[imageIdx + 2] = tempRGB;
        }
    }

    void save_as_bmp(char* filename)
    {
        BITMAPFILEHEADER bitmapFileHeader;
        memset(&bitmapFileHeader, 0, sizeof(BITMAPFILEHEADER));
        bitmapFileHeader.bfSize = sizeof(BITMAPFILEHEADER);
        bitmapFileHeader.bfType = 0x4d42;	//BM
        bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

        // заполнить bitmapinfoheader.
        BITMAPINFOHEADER bitmapInfoHeader;
        memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
        bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmapInfoHeader.biWidth = x;
        bitmapInfoHeader.biHeight = y;
        bitmapInfoHeader.biPlanes = 1;
        bitmapInfoHeader.biBitCount = 32;
        bitmapInfoHeader.biCompression = BI_RGB;
        bitmapInfoHeader.biSizeImage = x * abs(y) * 4;

        //////////////////////////////////////////////////////////////////////////
        FILE* filePtr;
        int imageIdx;
        unsigned char tempRGB;

        //swap R B
        for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 4)
        {
            tempRGB = pixels[imageIdx];
            pixels[imageIdx] = pixels[imageIdx + 2];
            pixels[imageIdx + 2] = tempRGB;
        }

        filePtr = fopen(filename, "wb");

        fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

        fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

        fwrite(pixels, bitmapInfoHeader.biSizeImage, 1, filePtr);

        fclose(filePtr);

        //swap R B again
        for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 4)
        {
            tempRGB = pixels[imageIdx];
            pixels[imageIdx] = pixels[imageIdx + 2];
            pixels[imageIdx + 2] = tempRGB;
        }
        
    }

    void display_and_exit( void(*e)(void*) = NULL ) 
    {
        CPUBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        bitmapExit = e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = "";
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
        glutInitWindowSize( x, y );
        glutCreateWindow( "bitmap" );
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        glutMainLoop();
    }

     // static method used for glut callbacks
    static CPUBitmap** get_bitmap_ptr( void ) {
        static CPUBitmap   *gBitmap;
        return &gBitmap;
    }

   // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27:
                CPUBitmap*   bitmap = *(get_bitmap_ptr());
                if (bitmap->dataBlock != NULL && bitmap->bitmapExit != NULL)
                    bitmap->bitmapExit( bitmap->dataBlock );
                exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw( void ) {
        CPUBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
        glFlush();
    }
};

#endif  // __CPU_BITMAP_H__
