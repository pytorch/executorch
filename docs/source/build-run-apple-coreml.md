<!---- To make this progress bar work users will need to modify source/_templates/layout.html >
<!---- To make this page show up in the tutorials section users will need to add an entry in source/index.rst under the Tutorials section>

<!---- DO NOT MODIFY Progress Bar Start --->

<div class="progress-bar-wrapper">
   <div class="progress-bar-item">
     <div class="step-number" id="step-1">1</div>
     <span class="step-caption" id="caption-1"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-2">2</div>
     <span class="step-caption" id="caption-2"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-3">3</div>
     <span class="step-caption" id="caption-3"></span>
   </div>
   <div class="progress-bar-item">
     <div class="step-number" id="step-4">4</div>
     <span class="step-caption" id="caption-4"></span>
   </div>
</div>

<!---- DO NOT MODIFY Progress Bar End--->

# TITLE

<!----This will show a grid card on the page----->
::::{grid} 2
:::{grid-item-card}  What you will learn in this tutorial:
:class-card: card-learn
* In this tutorial you will learn how to lower and deploy a model for Apple devices using CoreML Delegate.
:::
:::{grid-item-card}  Tutorials we recommend you complete before this:
:class-card: card-prerequisites
* [Introduction to ExecuTorch](intro-how-it-works.md)
* [Setting up ExecuTorch](getting-started-setup.md)
* [Building ExecuTorch with CMake](runtime-build-and-cross-compilation.md)
:::
::::

## Prerequsites (Hardware and Software)

Provide instructions on what kind of hardware and software are pre-requisite for the tutorial.

### Hardware:
 - Hardware requirements to go through this tutorial

### Software:
 - Software requirements to go through this tutorial

## Setting up your developer environment

Steps that the users need to go through to setup their developer environment for this tutorial.

## Build

### AOT (Ahead-of-time) components:

Describe what set of steps user should go through, as part of this tutorial, in order to export a model and ready it for execution on the target platform. Such steps, illustrated via example API invocations, may include quantization, delegation, custom passes, custom memory planning etc.

### Runtime:

Steps that the users need to go through to build the runtime/supporting app that they can then run on their target device.

The tutorial may target a) building separate executable via which a exported model can be run, or b) building libraries that should be linked into native apps available inside examples folder. You are free to add both options. For the latter, you can link to demo-app tutorial page, that should contain instructions on linking custom libraries.

## Deploying and running on device

Steps that the users need to go through to deploy and run the runtime and model generated from the previous steps.

## Frequently encountered errors and resolution.

Describe what common errors uses may see and how to resolve them.

If you encountered any bugs or issues following this tutorial please file a bug/issue here at XYZ, with hashtag #ExecuTorch #MyHashTag..
