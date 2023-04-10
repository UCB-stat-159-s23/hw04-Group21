import numpy as np
from ligotools import readligo as rl
import pytest
import json

fnjson = "data/BBH_events_v3.json"
events = json.load(open(fnjson,"r"))
eventname  = 'GW150914' 
event = events[eventname]
fn_H1 = event['fn_H1']          
fn_L1 = event['fn_L1']  

strain, time, chan_dict = rl.loaddata("data/L-L1_LOSC_4_V2-1126259446-32.hdf5", 'L1')

def test_loaddata():
	"""
	test the format of strain, time, and chan_dict
	"""
	assert isinstance(strain,np.ndarray), "STRAIN is not a numpy vector."
	assert isinstance(time,np.ndarray) , "TIME is not a numpy vector."
	assert isinstance(chan_dict, dict), "CHANNEL_DICT is not a dictionary."


def test_dq_channel_to_seglist():
	"""
	test length of seglist
	"""
	DQflag1 = 'CBC_CAT3'
	segment_list1 = rl.dq_channel_to_seglist(chan_dict[DQflag1])
	assert len(segment_list1)==1, "segment list should be length 1."
	
	DQflag2 = 'NO_CBC_HW_INJ'
	segment_list2 = rl.dq_channel_to_seglist(chan_dict[DQflag2])
	assert len(segment_list2)==1, "segment list should be length 1."


def test_sizes():
	"""
	test size of strain, time, and channel_dict
	"""
	assert np.size(strain)==131072, "STRAIN is not of the correct size"
	assert np.size(time)==131072, "TIME is not of the correct size"
	assert np.size(chan_dict)==1, "CHANNEL_DICT is not of the correct size"
	

def test_dq2segs():
	"""
	test content of the seglist
	"""
	seglist_chan_dict1 = rl.dq2segs(chan_dict, 1).seglist
	seglist_chan_dict2 = rl.dq2segs(chan_dict, 2).seglist
	seglist_chan_dict3 = rl.dq2segs(chan_dict, 3).seglist
	assert seglist_chan_dict1==[(1, 33)], "seglist when start time = 1 is not correct"
	assert seglist_chan_dict2==[(2, 34)], "seglist when start time = 2 is not correct"
	assert seglist_chan_dict3==[(3, 35)], "seglist when start time = 3 is not correct"
