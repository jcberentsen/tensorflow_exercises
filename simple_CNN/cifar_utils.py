import cPickle
import numpy as np

def load_batch(filename):
	fo = open(filename, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def extract_plane(raw, index):
	data = np.asarray(raw[index:index+1024], dtype=np.uint8)
	return np.reshape(data, (32, 32))

def shape_image(raw):
	# raw = np.asarray(batch_item, dtype=np.uint8)
	red = extract_plane(raw,0)
	green = extract_plane(raw, 1024)
	blue = extract_plane(raw, 2048)
	return np.dstack((red, green, blue))

def	load_batch_details(batch_filename):
	batch = load_batch(batch_filename)
	return map(shape_image, batch['data']), batch['labels']