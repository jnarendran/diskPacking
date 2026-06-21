# diskPacking
Visualization for packing disks on a spherical surface with and without a defect and visualizing force chains.

The file *percolation_tester_FC_visualization.py* takes in an input for number of grains, touch threshold distance, and hemispherical shell parameters to output an interactive visualization of force chains, number of force chains, and the number of grains participating in the force chains.

![700 disks arranged on a hemispherical surface](/img/700grains.png)
Here is an example showing 700 disks packed on a hemispherical surface with the highlighted disks being part of force chain networks.


The file *percolation_tester_defect.py* has the inputs and outputs as above and also takes into account granular vacancy defect size.  

Both can also give you force chain and grain data for a range of grain values or defect sizes for a given number of simulation trials.
