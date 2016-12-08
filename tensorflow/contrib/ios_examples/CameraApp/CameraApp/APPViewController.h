//
//  APPViewController.h
//  CameraApp
//
//  Created by Rafael Garcia Leiva on 10/04/13.
//  Copyright (c) 2013 Appcoda. All rights reserved.
//

#import <UIKit/UIKit.h>

#include <memory>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/memmapped_file_system.h"

@interface APPViewController : UIViewController <UIImagePickerControllerDelegate, UINavigationControllerDelegate> {
    NSMutableArray *labelLayers;
    std::unique_ptr<tensorflow::Session> tf_session;
    std::unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;
    std::vector<std::string> labels;
    bool show_alert;
}

@property (strong, nonatomic) IBOutlet UIImageView *imageView;

- (IBAction)takePhoto:  (UIButton *)sender;
- (IBAction)selectPhoto:(UIButton *)sender;

@end
