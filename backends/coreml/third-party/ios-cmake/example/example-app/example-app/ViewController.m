//
//  ViewController.m
//  example-app
//
//  Created by Alexander Widerberg on 2017-02-03.
//  Copyright © 2017 example. All rights reserved.
//

#import "ViewController.h"
#import "HelloWorldIOS.h"

@interface ViewController ()
@property (weak, nonatomic) IBOutlet UILabel *helloWorldLabel;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    HelloWorldIOS *helloWorld = [HelloWorldIOS new];
    self.helloWorldLabel.text = [helloWorld getHelloWorld];
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
