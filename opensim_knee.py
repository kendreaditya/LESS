# https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python#ScriptinginPython-Manualinstallation(OpenSim4.3orlater)
import opensim as osim

# Load the model
model = osim.Model('knee_model.osim')
state = model.initSystem()

# Create a time series table for joint angles
import pandas as pd
from opensim import TimeSeriesTable

data = {
    'time': time_data,
    'knee_angle': knee_angle_data,
    'hip_angle': hip_angle_data,
    # Add other joint angles as needed
}

df = pd.DataFrame(data)
table = TimeSeriesTable(df)

# Create a states trajectory from the table
from opensim import StatesTrajectory
states_trajectory = StatesTrajectory.createFromStatesTable(model, table)

# Apply the states trajectory to the model
manager = osim.Manager(model)
manager.initialize(state)
manager.integrate(state, time_data[-1])

# Analyze ligament forces
force_reporter = osim.ForceReporter()
model.addAnalysis(force_reporter)
model.realizeDynamics(state)

# Get the results
results = force_reporter.getForcesTable()
