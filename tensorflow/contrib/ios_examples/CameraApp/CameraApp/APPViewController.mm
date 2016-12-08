//
//  APPViewController.m
//  CameraApp
//
//  Created by Rafael Garcia Leiva on 10/04/13.
//  Copyright (c) 2013 Appcoda. All rights reserved.
//

#import "APPViewController.h"

#import <QuartzCore/QuartzCore.h>
//#include <sys/time.h>

#include "ios_image_load.h"
#include "tensorflow_utils.h"
//#define NSLog(FORMAT, ...) printf("%s\n", [[NSString stringWithFormat:FORMAT, ##__VA_ARGS__] UTF8String]);

// If you have your own model, modify this to the file name, and make sure
// you've added the file to your app resources too.
static NSString* model_file_name = @"mmapped_v2";
static NSString* model_file_type = @"pb";
// This controls whether we'll be loading a plain GraphDef proto, or a
// file created by the convert_graphdef_memmapped_format utility that wraps a
// GraphDef and parameter file that can be mapped into memory from file to
// reduce overall memory usage.
const bool model_uses_memory_mapping = true;
// If you have your own model, point this to the labels file.
static NSString* labels_file_name = @"labels";
static NSString* labels_file_type = @"txt";
// These dimensions need to match those the model was trained with.
const int wanted_input_width = 299;
const int wanted_input_height = 299;
const int wanted_input_channels = 3;
const std::string input_layer_name = "original_input";
const std::string output_layer_name = "Softmax_1";
const bool logging = YES;


@interface APPViewController ()

@end

@implementation APPViewController

- (void)viewDidLoad {
    
    [super viewDidLoad];
    
    // label Layers to output predictions
    labelLayers = [[NSMutableArray alloc] init];
    
    // load tensorflow graph
    tensorflow::Status load_status;
    if (model_uses_memory_mapping) {
        load_status = LoadMemoryMappedModel(
                                            model_file_name, model_file_type, &tf_session, &tf_memmapped_env);
    } else {
        load_status = LoadModel(model_file_name, model_file_type, &tf_session);
    }
    if (!load_status.ok()) {
        LOG(FATAL) << "Couldn't load model: " << load_status;
    }
    
    // load labels
    tensorflow::Status labels_status =
    LoadLabels(labels_file_name, labels_file_type, &labels);
    if (!labels_status.ok()) {
        LOG(FATAL) << "Couldn't load labels: " << labels_status;
    }
    show_alert = YES;
    
    
}

- (void)viewDidAppear:(BOOL)animated {
    
    [super viewDidAppear:animated];
    
    // alert if no camera
    if (![UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera] & show_alert) {
        UIAlertController* alert = [UIAlertController
                                    alertControllerWithTitle:@"Error"
                                    message:@"Device has no camera"
                                    preferredStyle:UIAlertControllerStyleAlert
                                    ];
        
        UIAlertAction* defaultAction = [UIAlertAction
                                        actionWithTitle:@"OK"
                                        style:UIAlertActionStyleDefault
                                        handler:^(UIAlertAction * action) {}];
        
        [alert addAction:defaultAction];
        
        [self presentViewController:alert animated:YES completion:nil];
        show_alert = NO;
        
    }
    NSLog(@"viewDidAppear");

}

- (void)viewDidUnload {
    
    [super viewDidUnload];
    NSLog(@"viewDidUnload");
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [self removeAllLabelLayers];
    NSLog(@"viewWillDisappear");

}
- (IBAction)takePhoto:(UIButton *)sender {
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    picker.delegate = self;
    picker.allowsEditing = NO;
    picker.sourceType = UIImagePickerControllerSourceTypeCamera;
    
    [self presentViewController:picker animated:YES completion:NULL];
    
}

- (IBAction)selectPhoto:(UIButton *)sender {
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    picker.delegate = self;
    picker.allowsEditing = NO;
    picker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
    
    [self presentViewController:picker animated:YES completion:NULL];
    
}

#pragma mark - Image Picker Controller delegate methods

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary *)info {
    UIImage *chosenImage = [info objectForKey:UIImagePickerControllerOriginalImage];
    
    [self.imageView setImage: chosenImage];
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(void) {
        [self runCNNOnFrame: chosenImage];
    });
    [picker dismissViewControllerAnimated:YES completion:NULL];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {

    [picker dismissViewControllerAnimated:YES completion:NULL];
    NSLog(@"imagePickerControllerDidCancel");
    
}

- (void)runCNNOnFrame:(UIImage*)img {
    assert(img != NULL);
    
    int image_width;
    int image_height;
    int image_channels;
    unsigned char *sourceStartAddr;
    sourceStartAddr = getRGBAsFromImage(img, &image_width, &image_height, &image_channels);
    
    assert(image_channels >= wanted_input_channels);
    tensorflow::Tensor image_tensor(
                                    tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape(
                                                            {1, image_height, image_width, wanted_input_channels}
                                                            )
                                    );
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    tensorflow::uint8 *in = sourceStartAddr;
    float *out = image_tensor_mapped.data();
    
    // Log directories
    NSString *documentsDirectory = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents"];
    NSString *filename = [documentsDirectory stringByAppendingPathComponent:[NSString stringWithFormat:@"log%i.txt", (int)[[NSDate date] timeIntervalSinceReferenceDate]]];
    
    NSString *log_str;
    NSFileHandle *file;
    if (logging == YES) {
        [[NSFileManager defaultManager] createFileAtPath:filename contents:nil attributes:nil];
        NSFileManager *filemgr = [NSFileManager defaultManager];
        if ([filemgr fileExistsAtPath: filename ] == NO)
            NSLog (@"File %@ not found", filename);
        file = [NSFileHandle fileHandleForUpdatingAtPath:filename];
        log_str = @"";
    }
    for (int y = 0; y < image_height; ++y) {
        float *out_row = out + (y * image_width * wanted_input_channels);
        for (int x = 0; x < image_width; ++x) {
            const int in_x = x;
            const int in_y = y;
            tensorflow::uint8 *in_pixel =
            in + (in_y * image_width * image_channels) + (in_x * image_channels);
            float *out_pixel = out_row + (x * wanted_input_channels);
            for (int c = 0; c < wanted_input_channels; ++c) {
                out_pixel[c] = in_pixel[c];
                if (logging == YES) {
                    log_str = [log_str stringByAppendingString:[NSString stringWithFormat:@"%u,", in_pixel[c]]];
                    [file seekToEndOfFile];
                    [file writeData: [log_str dataUsingEncoding:NSUTF8StringEncoding]];
                    log_str = @"";
                }
            }
        }
    }
    if (logging == YES) {
        log_str = @"\nPrediction";
    }
    if (tf_session.get()) {
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = tf_session->Run(
                                                        {{input_layer_name, image_tensor}}, {output_layer_name}, {}, &outputs
                                                        );
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed:" << run_status;
        } else {
            NSLog(@"predictions retrieved");
            tensorflow::Tensor *output = &outputs[0];
            auto predictions = output->flat<float>();
            
            NSMutableDictionary *newValues = [NSMutableDictionary dictionary];
            for (int index = 0; index < predictions.size(); index += 1) {
                const float predictionValue = predictions(index);
                if (logging == YES) {
                    log_str = [log_str stringByAppendingString:[NSString stringWithFormat:@",%f", predictionValue]];
                }
                if (predictionValue > 0.05f) {
                    std::string label = labels[index % predictions.size()];
                    NSString *labelObject = [NSString stringWithCString:label.c_str() encoding:NSASCIIStringEncoding];
                    NSNumber *valueObject = [NSNumber numberWithFloat:predictionValue];
                    [newValues setObject:valueObject forKey:labelObject];
                }
            }
            dispatch_async(dispatch_get_main_queue(), ^(void) {
                [self setPredictionValues:newValues];
            });
        }
    }
    if (logging == YES) {
        [file seekToEndOfFile];
        [file writeData: [log_str dataUsingEncoding:NSUTF8StringEncoding]];
        [file closeFile];
    }
}

- (void)setPredictionValues:(NSDictionary *)newValues {
    
    NSArray *candidateLabels = [NSMutableArray array];
    for (NSString *label in newValues) {
        NSNumber *oldPredictionValueObject =
        [newValues objectForKey:label];
        const float oldPredictionValue = [oldPredictionValueObject floatValue];
        if (oldPredictionValue > 0.05f) {
            NSDictionary *entry = @{
                                    @"label" : label,
                                    @"value" : oldPredictionValueObject
                                    };
            candidateLabels = [candidateLabels arrayByAddingObject:entry];
        }
    }
    NSSortDescriptor *sort =
    [NSSortDescriptor sortDescriptorWithKey:@"value" ascending:NO];
    NSArray *sortedLabels = [candidateLabels
                             sortedArrayUsingDescriptors:[NSArray arrayWithObject:sort]];
    
    const float leftMargin = 10.0f;
    const float topMargin = 30.0f;
    
    const float valueWidth = 48.0f;
    const float valueHeight = 26.0f;
    
    const float labelWidth = 246.0f;
    const float labelHeight = 26.0f;
    
    const float labelMarginX = 5.0f;
    const float labelMarginY = 5.0f;
    
    int labelCount = 0;
    for (NSDictionary *entry in sortedLabels) {
        NSString *label = [entry objectForKey:@"label"];
        NSNumber *valueObject = [entry objectForKey:@"value"];
        const float value = [valueObject floatValue];
        
        const float originY =
        (topMargin + ((labelHeight + labelMarginY) * labelCount));
        
        const int valuePercentage = (int)roundf(value * 100.0f);
        
        const float valueOriginX = leftMargin;
        NSString *valueText = [NSString stringWithFormat:@"%d%%", valuePercentage];
        
        [self addLabelLayerWithText:valueText
                            originX:valueOriginX
                            originY:originY
                              width:valueWidth
                             height:valueHeight
                          alignment:kCAAlignmentRight];
        
        const float labelOriginX = (leftMargin + valueWidth + labelMarginX);
        
        [self addLabelLayerWithText:[label capitalizedString]
                            originX:labelOriginX
                            originY:originY
                              width:labelWidth
                             height:labelHeight
                          alignment:kCAAlignmentLeft];
        
        labelCount += 1;
        if (labelCount > 4) {
            break;
        }
    }
}

- (void)removeAllLabelLayers {
    for (CATextLayer *layer in labelLayers) {
        [layer removeFromSuperlayer];
    }
    [labelLayers removeAllObjects];
}

- (void)addLabelLayerWithText:(NSString *)text
                      originX:(float)originX
                      originY:(float)originY
                        width:(float)width
                       height:(float)height
                    alignment:(NSString *)alignment {
    NSString *const font = @"Menlo-Regular";
    const float fontSize = 20.0f;
    
    const float marginSizeX = 5.0f;
    const float marginSizeY = 2.0f;
    
    const CGRect backgroundBounds = CGRectMake(originX, originY, width, height);
    
    const CGRect textBounds =
    CGRectMake((originX + marginSizeX), (originY + marginSizeY),
               (width - (marginSizeX * 2)), (height - (marginSizeY * 2)));
    
    CATextLayer *background = [CATextLayer layer];
    [background setBackgroundColor:[UIColor blackColor].CGColor];
    [background setOpacity:0.5f];
    [background setFrame:backgroundBounds];
    background.cornerRadius = 5.0f;
    
    [[self.view layer] addSublayer:background];
    [labelLayers addObject:background];
    
    CATextLayer *layer = [CATextLayer layer];
    [layer setForegroundColor:[UIColor whiteColor].CGColor];
    [layer setFrame:textBounds];
    [layer setAlignmentMode:alignment];
    [layer setWrapped:YES];
    [layer setFont:font];
    [layer setFontSize:fontSize];
    layer.contentsScale = [[UIScreen mainScreen] scale];
    [layer setString:text];
    
    [[self.view layer] addSublayer:layer];
    [labelLayers addObject:layer];
}

@end
