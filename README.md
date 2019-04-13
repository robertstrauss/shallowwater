# Tsunami Forecasting and Risk Analysis
Tsunamis are often tragic events, killing thousands at a time and, because of their
large range, often come with no warning. In Indonesia alone, there have been two tsunamis in
the last six months, each killing hundreds of people. How can the distribution of damage from
a tsunami be forecasted? This is already done with supercomputers, but this isn’t practical for
an early warning system and isn’t affordable to small agencies or villages in developing
tsunami-prone areas. In this project, an accurate, affordable, real time tsunami forecasting
system was created. This model was validated with two test cases: a tsunami near Palu,
Indonesia, and a tsunami from Krakatau volcano, Indonesia.
I constructed a computer model of the ocean from scratch in over a thousand lines of
code. This implemented the non-linear coupled partial differential shallow water equations. I
achieved faster than real time simulation by parallelization on an inexpensive graphics card,
as opposed to a supercomputer. I validated the forecast of my model on the 2018 Palu and
2019 Krakatau events. My model predicted with high accuracy not only where the high
waves were, but also where the low waves were. This is important because people may not
know where to flee to or to stay put.
The simulations also show what physical factors contribute to the severity. My
findings show: the shape of the bay geometry around Palu City was important, and the initial
location was critical. Other tsunami properties were identified as less important factors.
