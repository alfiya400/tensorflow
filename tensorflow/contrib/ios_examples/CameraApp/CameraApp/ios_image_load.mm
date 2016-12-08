#include "ios_image_load.h"

unsigned char* getRGBAsFromImage(UIImage* img,
				     int* out_width, int* out_height,
				     int* out_channels) {
    int orig_width = (int)img.size.width;
    int orig_height = (int)img.size.height;
    
    // resize image so that max edge is 299
    float scale = 299. / (orig_width > orig_height ? orig_width : orig_height);
    UIImage* resizedImage = [UIImage imageWithCGImage:img.CGImage scale: (1. / scale) orientation:img.imageOrientation];
    int width = (int)resizedImage.size.width;
    int height = (int)resizedImage.size.height;
    UIImageOrientation orientation = resizedImage.imageOrientation;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    unsigned char *rawData = (unsigned char*) calloc(width * height * 4, sizeof(unsigned char));
    int bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;

    // copied from http://stackoverflow.com/a/5427890
    // We need to calculate the proper transformation to make the image upright.
    // We do it in 2 steps: Rotate if Left/Right/Down, and then flip if Mirrored.
    CGAffineTransform transform = CGAffineTransformIdentity;
    
    switch (orientation) {
        case UIImageOrientationDown:
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformTranslate(transform, width, height);
            transform = CGAffineTransformRotate(transform, M_PI);
            break;
            
        case UIImageOrientationLeft:
        case UIImageOrientationLeftMirrored:
            transform = CGAffineTransformTranslate(transform, width, 0);
            transform = CGAffineTransformRotate(transform, M_PI_2);
            break;
            
        case UIImageOrientationRight:
        case UIImageOrientationRightMirrored:
            transform = CGAffineTransformTranslate(transform, 0, height);
            transform = CGAffineTransformRotate(transform, -M_PI_2);
            break;
        case UIImageOrientationUp:
        case UIImageOrientationUpMirrored:
            break;
    }
    
    switch (orientation) {
        case UIImageOrientationUpMirrored:
        case UIImageOrientationDownMirrored:
            transform = CGAffineTransformTranslate(transform, width, 0);
            transform = CGAffineTransformScale(transform, -1, 1);
            break;
            
        case UIImageOrientationLeftMirrored:
        case UIImageOrientationRightMirrored:
            transform = CGAffineTransformTranslate(transform, height, 0);
            transform = CGAffineTransformScale(transform, -1, 1);
            break;
        case UIImageOrientationUp:
        case UIImageOrientationDown:
        case UIImageOrientationLeft:
        case UIImageOrientationRight:
            break;
    }
    
    // Now we draw the underlying CGImage into a new context, applying the transform
    // calculated above.
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    if (orientation != UIImageOrientationUp) {
        CGContextConcatCTM(context, transform);
    }
    
    switch (resizedImage.imageOrientation) {
        case UIImageOrientationLeft:
        case UIImageOrientationLeftMirrored:
        case UIImageOrientationRight:
        case UIImageOrientationRightMirrored:
            CGContextDrawImage(context, CGRectMake(0,0,height,width), resizedImage.CGImage);
            break;
            
        default:
            CGContextDrawImage(context, CGRectMake(0,0,width,height), resizedImage.CGImage);
            break;
    }
    
    
    CGContextRelease(context);

    *out_width = width;
    *out_height = height;
    *out_channels = bytesPerPixel;
    NSLog(@"Orig - %i %i, resized - %i %i", orig_width, orig_height, width, height);
    return rawData;
}
