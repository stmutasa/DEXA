"""
Pre processes all of the localization inputs and saves as tfrecords
Also handles loading tfrecords

The standard we're using is saving the top left corner and bottom right corner in numpy format
which is rows x columns or y,x: [ymin, xmin, ymax, xmax]
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path
import json
import os
import Utils

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = '/data/Datasets/DEXA/'

rawCR_dir = home_dir + 'Raw/'
test_loc_folder = home_dir + 'Test/'
cleanCR_folder = home_dir + 'Cleaned_CR/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()


def pre_proc_localizations(box_dims=64, thresh=0.6):

    """
    Pre processes the input for the localization network
    :param box_dims: dimensions of the saved images
    :return:
    # TODO: To make classificaton negatives, use large stride and adjust IOU for exclusion to 1.0
    """

    group = 'Box_Locs'

    # Load the files and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, cleanCR_folder)
    dexa_scores = sdl.load_CSV_Dict('Mrn', 'Dexa_Scores.csv')
    gtboxes = sdl.load_CSV_Dict('filename', 'Dexa_gtboxes.csv')

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0
    heights, weights = [], []
    wards, lap_count = 0, 0

    for file in filenames:

        # Load the file only first
        try:
            image, _, _, photometric, header = sdl.load_DICOM_2D(file)
            if photometric == 1: image *= -1
        except Exception as e:
            print('DICOM Error: %s' % e)
            continue

        # Retreive the patient information from the saved file: E01279372_L-HIP_V1-1
        base = os.path.basename(file).split('.')[0].split('_')
        acc, laterality, part = base

        # Get the MRN, this is how we match the DEXA to the Hip radiograph
        mrn = int(header['tags'].PatientID)
        mrn = str(mrn)

        # Get the dexa scores
        try: densities = dexa_scores[mrn]
        except Exception as e:
            print('Dexa score error: ', e)
            continue

        # We want to use the average of the available scores
        _tot, _sum = 0, 0

        # Loop and retreive values only if they exist
        for i, v in densities.items():
            if v and i != 'Acc':
                _sum += float(v)
                _tot += 1

        # Now collect the average, some have no densities saved
        try: density = _sum/_tot
        except: continue

        # Now retreive the ground truth boxes
        dst_File = os.path.basename(file).split('.')[0] + '.png'
        try: annotations = gtboxes[dst_File]
        except Exception as e:
            print ('Retreive gtbox error: ', e)
            continue

        # Some patients have multiple bounding boxes, retreive region count
        region_count, region_id = int(annotations['region_count']), annotations['region_id']

        # Keep track of all the regions obtained in a list
        regions_parsed, _gtboxes = [], []
        regions_parsed.append(region_id)

        # Convert the first found region string to a dictionary then add it to the list
        box = json.loads(annotations['region_shape_attributes'])
        if not box: continue
        _gtboxes.append(box)

        # If there are more than one regions, get the others, otherwise we are good
        if region_count >1:

            # Loop through all the annotations and add the rest of the gtboxes to the list
            for idx, dic in gtboxes.items():
                #print (idx,' --- ', dic['region_id'], ' --- ', dic['region_count'], ' --- ', dst_File)

                # If this region hasn't been added then add it to the tracking lists
                # TODO: Why the fuck is it adding a 1 to the index??
                if idx == dst_File + '1' or idx == dst_File:
                    if not dic['region_id'] in regions_parsed:

                        # Convert the string to a dictionary
                        _box = json.loads(dic['region_shape_attributes'])
                        if not _box: continue

                        # Add to the lists
                        _gtboxes.append(_box)
                        regions_parsed.append(dic['region_id'])

        """ 
        Now we have a list of all the GT boxes for this image.
        if there are two, split the image in half and separate each region into the correct half 
        If there is one region, skip the splitting part
        Then make the anchor boxes     
        """

        # Make dummy arrays to hold the images and append the original
        images = [image]

        # Two region case
        if region_count > 1:

            # Split the image in half by copying each half into a dummy array
            hi = image.shape[1]//2
            left_img, right_img = image[:, :hi], image[:, hi:hi*2]

            # Add to images tracker, replace image 0 with left image
            images[0] = left_img
            images.append(right_img)

            # Now we need to add the correct region to the correct box
            if _gtboxes[0]['x'] > _gtboxes[1]['x']:

                # Swap the list elements
                _gtboxes[0], _gtboxes[1] = _gtboxes[1], _gtboxes[0]

                # Remember we need to subtract half the image width from X in the right image coordinates
                _gtboxes[1]['x'] -= hi

            else:

                # Default case, keep index 0 in place but subtract image width from index 1's x coordinate
                _gtboxes[1]['x'] = _gtboxes[1]['x'] - hi

        # Now just loop through the storage list and make our anchors!!
        for i in range(region_count):

            # Get the right gtbox
            gtbox = _gtboxes[i]

            # Normalize the image we are working on
            image = sdl.adaptive_normalization(images[i]).astype(np.float32)

            """
            Generate the anchor boxes here. Here are the stats:
            510 bounding boxes. H/W AVG: 318.200/293.647 Max: 568.000/632.000 STD: 59.444/56.600
            510 Norm bounding boxes. H/W AVG: 0.142/0.130 Max: 0.249/0.290 STD: 0.023/0.034
            510 Images. H/W AVG: 2268.8/2343.8 Max: 3172.0/3546.0 STD: 379.345/454.546
            510 Ratios. Max: 1.81, Min: 0.70, Avg: 1.10 STD: 0.183
            """

            # Avg height, width, ratio. STD of ratio and scale
            sh, sw, rat, ratSD, scaSD = 318.2, 293.647, 1.1, 0.196 * 1.25, (59.444 / 318.2 + 56.600 / 293.647) / 2
            anchors = Utils.generate_anchors(image, [sh, sw], 16, ratios=[rat - ratSD, rat, rat + ratSD],
                                             scales=[1 - scaSD, 1.0, 1 + scaSD])

            # Generate a GT box measurement in corner format
            ms = image.shape
            gtbox = np.asarray([gtbox['y'], gtbox['x'], gtbox['y'] + gtbox['height'], gtbox['x'] + gtbox['width'],
                                gtbox['y'] + (gtbox['height'] / 2), gtbox['x'] + (gtbox['width'] / 2), gtbox['height'],
                                gtbox['width']])

            # Append IOUs
            IOUs = Utils._iou_calculate(anchors[:, :4], gtbox[:4])
            if np.max(IOUs) <= thresh // 2:
                del image, anchors, gtbox
                continue
            anchors = np.append(anchors, IOUs, axis=1)

            # Normalize the GT boxes
            norm_gtbox = np.asarray([gtbox[0] / ms[0], gtbox[1] / ms[1],
                                     gtbox[2] / ms[0], gtbox[3] / ms[1],
                                     gtbox[4] / ms[0], gtbox[5] / ms[1],
                                     gtbox[6] / ms[0], gtbox[7] / ms[1],
                                     ms[0], ms[1]]).astype(np.float32)

            # Generate boxes by looping through the anchor list
            for an in anchors:

                # Remember anchor = [10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU]
                an = an.astype(np.float32)

                # Anchors between IOU of 0.25 and 0.6 do not contribute:
                if an[8] > 0.3 and an[8] < thresh:
                    del an
                    continue

                # Generate a box at this location
                anchor_box, _ = sdl.generate_box(image, an[4:6].astype(np.int16), an[6:8].astype(np.int16), dim3d=False)

                # Reshape the box to a standard dimension: 128x128
                anchor_box = sdl.zoom_2D(anchor_box, [box_dims, box_dims]).astype(np.float16)

                # Norm the anchor box dimensions
                anchor = [
                    an[0] / ms[0], an[1] / ms[1], an[2] / ms[0], an[3] / ms[1], an[4] / ms[0], an[5] / ms[1],
                    an[6] / ms[0], an[7] / ms[1], an[8]
                ]

                # Append the anchor to the norm box
                box_data = np.append(norm_gtbox, anchor)

                # Make object and fracture labels = 1 if IOU > threhsold IOU
                fracture_class = 0
                if box_data[-1] >= thresh:
                    object_class = 1
                else:
                    object_class = 0
                counter[object_class] += 1

                # Append object class and fracture class to box data
                box_data = np.append(box_data, [object_class, fracture_class]).astype(np.float32)

                # Save the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
                #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, IOU, obj_class, #_class]
                data[index] = {'data': anchor_box, 'box_data': box_data, 'group': group, 'view': dst_File,
                               'accno': acc}

                # Increment box count
                index += 1

                # Garbage
                del anchor_box, an, box_data

            # TODO: Dummy tfrecords Dir
            tfrecords_dir = 'data'

            # Save q xx patients
            saveq = 50
            if pt % saveq == 0 and pt != 0:
                print('\nMade %s (%s) bounding boxes SO FAR from %s patients. %s Positive and %s Negative (%.6f %%)'
                      % (index, index - lap_count, pt, counter[1], counter[0], 100 * counter[1] / index))
                lap_count = index

                if pt < (saveq + 5):
                    sdl.save_dict_filetypes(data[0])
                    sdl.save_tfrecords(data, 1, file_root=('%s/test/BOX_LOCS%s' % (tfrecords_dir, pt // saveq)))
                else:
                    sdl.save_tfrecords(data, 1, file_root=('%s/train/BOX_LOCS%s' % (tfrecords_dir, pt // saveq)))

                del data
                data = {}

            # Increment hip counter
            pt += 1
            del image, anchors, gtbox

        # Increment patient counter
        wards += 1
        del images, _gtboxes

    # Save the data.
    sdl.save_tfrecords(data, 1, file_root=('%s/train/BOX_LOCSFin' % tfrecords_dir))

    # Done with all patients
    print('\nDone, made %s bounding boxes from %s hips in %s patients. %s Positive and %s Negative (%.6f %%)'
          % (index, wards, pt, counter[1], counter[0], counter[1] / index))


def load_protobuf_loc(training=True):

    """
    Loads the protocol buffer into a form to send to shuffle. To oversample classes we made some mods...
    Load with parallel interleave -> Prefetch -> Large Shuffle -> Parse labels -> Undersample map -> Flat Map
    -> Prefetch -> Oversample Map -> Flat Map -> Small shuffle -> Prefetch -> Parse images -> Augment -> Prefetch -> Batch
    """

    # Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
    #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

    # Lambda functions for retreiving our protobuf
    _parse_all = lambda dataset: sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float16,
                                                    'box_data', tf.float32, [21])

    # Load tfrecords with parallel interleave if training
    if training:
        filenames = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        files = tf.data.Dataset.list_files(os.path.join(FLAGS.data_dir, '*.tfrecords'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=len(filenames),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print('******** Loading Files: ', filenames)
    else:
        files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)
        print('******** Loading Files: ', files)

    # Shuffle and repeat if training phase
    if training:

        # Define our undersample and oversample filtering functions
        _filter_fn = lambda x: sdl.undersample_filter(x['box_data'][19], actual_dists=[0.999, 0.00156], desired_dists=[.9, .1])
        _undersample_filter = lambda x: dataset.filter(_filter_fn)
        _oversample_filter = lambda x: tf.data.Dataset.from_tensors(x).repeat(
            sdl.oversample_class(x['box_data'][19], actual_dists=[0.9982, 0.001755], desired_dists=[.9, .1]))

        # Large shuffle, repeat for xx epochs then parse the labels only
        dataset = dataset.shuffle(buffer_size=int(5e5))
        dataset = dataset.repeat(FLAGS.repeats)
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Now we have the labels, undersample then oversample.
        dataset = dataset.map(_undersample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)
        dataset = dataset.map(_oversample_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: x)

        # Now perform a small shuffle in case we duplicated neighbors, then prefetch before the final map
        dataset = dataset.shuffle(buffer_size=FLAGS.batch_size)

    else: dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    if training: dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    else: dataset = dataset.batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Return the iterator
    return iterator


def load_protobuf_class(training=True):

    """
    Loads the classification network protobuf. No oversampling in this case
    """

    # Saved the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
    #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

    # Lambda functions for retreiving our protobuf
    _parse_all = lambda dataset: sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float16,
                                                    'box_data', tf.float16, [21])

    # Load tfrecords with parallel interleave if training
    if training:
        filenames = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        files = tf.data.Dataset.list_files(os.path.join(FLAGS.data_dir, '*.tfrecords'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=len(filenames),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print('******** Loading Files: ', filenames)
    else:
        files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)
        print('******** Loading Files: ', files)

    # Shuffle and repeat if training phase
    if training:

        # Large shuffle, repeat for xx epochs then parse the labels only
        dataset = dataset.shuffle(buffer_size=int(5e5))
        dataset = dataset.repeat(FLAGS.repeats)
        dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    else: dataset = dataset.map(_parse_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope):
        dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    if training: dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    else: dataset = dataset.batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Return data as a dictionary
    return iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

  def __init__(self, distords):
    self._distords = distords

  def __call__(self, record):

    """Process img for training or eval."""
    image = record['data']

    if self._distords:  # Training

        # Data Augmentation ------------------ Flip, Contrast, brightness, noise

        # Save the data to [0ymin, 1xmin, 2ymax, 3xmax, cny, cnx, 6height, 7width, 8origshapey, 9origshapex,
        #    10yamin, 11xamin, 12yamax, 13xamax, 14acny, 15acnx, 16aheight, 17awidth, 18IOU, 19obj_class, 20#_class]

        # Resize to network size
        image = tf.expand_dims(image, -1)
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)

        # Image rotation parameters
        angle = tf.random_uniform([], -0.26, 0.26)
        image = tf.contrib.image.rotate(image, angle)

        # Then randomly flip
        image = tf.image.random_flip_left_right(tf.image.random_flip_up_down(image))
        #image = tf.image.random_flip_left_right(image)

        # Random brightness/contrast
        image = tf.image.random_brightness(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.95, upper=1.05)

        # For noise, first randomly determine how 'noisy' this study will be
        T_noise = tf.random.uniform([], 0, 0.05)

        # Create a poisson noise array
        noise = tf.random.uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)

        # Normalize the image
        image = tf.image.per_image_standardization(image)
        image = tf.add(image, noise)

        # Add the poisson noise
        #image = tf.add(image, tf.cast(noise, tf.float16))

    else: # Validation

        image = tf.expand_dims(image, -1)

        # Normalize the image
        image = tf.image.per_image_standardization(image)

        # Resize to network size
        image = tf.image.resize_images(image, [FLAGS.network_dims, FLAGS.network_dims], tf.compat.v1.image.ResizeMethod.BICUBIC)

    # Make record image
    record['data'] = image

    return record
