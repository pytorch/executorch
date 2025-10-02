# module.yml

Do not remove this file. As mentioned in the official Zephyr [documenation](https://docs.zephyrproject.org/latest/develop/modules.html), for Executorch to be built as Zephyr module, the file `zephyr/module.yml` must exist at the top level directory in the project. 

# Work In Progress

We are currently working on request to the Zephyr project to formally support Executorch as a module. This will include an example of running executor runners on the Arm FVP, targetting the Zephyr RTOS. Once implemented, on executorch releases, the manifest in the Zephyr repo will need to be updated to point to the latest release of Executorch. More instructions on that will follow once the executorch module change is accepted into the Zephyr project.  
