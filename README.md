# atmospheric-glider-sim
Senior Design Simulation of a glider flight path 

Coding done by (Crypto Neutron., Zach M., and Jordan T.)

This simulation is to plan out an optimal flight path of a glider that has been deployed from a high altitude weather balloon for reaching a maximum capable height above a waypoint. The motivation behind the project is to create a more sustainable and accurate atmospheric profile compared to radiosondes. The code uses values from the excel spreadsheet provided to calculate various aircraft characteristics and forces during flight. 

The simulation has three options for gliding: steady state (no-wind), pre-defined wind profile (shown in text file), or forecasted wind data from the Global Forecasting System (GFS). The simulation is also able to output a waypoint file that contains: lat, lon, alt, and time spent in a certain regime. These waypoint files are formatted to be used in mission planner. The simulation can also output wind data which includes: altitude, wind speed, wind direction, and time in regime. A regime in these areas are defined by either a 0.5 degree box that contains a divided conditional box of GFS data or a descent of 200 ft (this value is hard coded).
