from compute_betti_curve import BC
import numpy as np
import os
import multiprocessing
import psutil
import itertools
import tqdm
import multiprocessing.pool


def symmetrization(x, data_format):

    """
        Symmetrize the feature maps in a given 4-D array.

        Parameters
        ----------
        x: 4-D array.
            The array need to be transfomred.
        
        data_format: "channel_first" or "channel_first".
            The data format of the array. 

        Returns
        -------
        x_symmetric: 4-D array.
            The 4-D array after processed.
        
    """

    x_symmetric = np.zeros(x.shape)
    if data_format == "channel_first":
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                arr = x[i, j, :, :]
                x_symmetric[i, j, :, :] = np.maximum(arr, np.transpose(arr))

    elif data_format == "channel_last":
        for i in range(x.shape[0]):
            for j in range(x.shape[-1]):
                arr = x[i, :, :, j]
                x_symmetric[i, :, :, j] = np.maximum(arr, np.transpose(arr))
    else:
        raise Exception("Data format Error! Please use channel_first or channel_last.")

    return x_symmetric


def birth_point(betti_curves, threshold = 50, betti_dimension = 0):

    """
        Calculate the birth point of a given Betti curve.

        Parameters
        ----------
        betti_curves: 2-D array.
            The given Betti curves yielded by a filtration with respect to feature topology.

        threshold: int.
            The threshold that determines the existence of birth point.
        
        betti_dimension: int.
            The dimension needs to be computed. If computeBetti0 is False, 1-Betti curve is the first column. Otherwise, 0-Betti curve is the first column.

        Returns
        -------
        birth point: int
            The birth point of a given Betti curve. Specifically, 0 denotes nonexistence. 

    """

    betti_array = betti_curves[:, betti_dimension]
    for i in range(betti_array.shape[0]):
        if i > threshold:
            return 0
        if betti_array[i] != 0:
            return i
    return 0
    

def process_SingleImage_pool(packed_params):
    symMatrix, filePrefix, workDirectory, baseDirectory = packed_params
    bc = BC(symMatrix, filePrefix = filePrefix, workDirectory = workDirectory, baseDirectory = baseDirectory)
    bettis = bc.compute_betti_curves()
    return bettis


def run_multiprocess_calculation(inputArrList, fileNameList, workDirectory, baseDirectory, num_worker):
    
    with multiprocessing.Pool(processes = num_worker) as p:
    # p = multiprocessing.pool.ThreadPool()
        ans = p.map(process_SingleImage_pool, zip(inputArrList, fileNameList, itertools.repeat(workDirectory), itertools.repeat(baseDirectory)))
    try:
        result = list(ans)
    except StopIteration:
        print("stop")
    except TimeoutError as error:
        print("function took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("function raised %s" % error)
        raise error
    # p.close()
    p.join()

    if len(psutil.Process(os.getpid()).children(recursive=True)) == 0:
        return result
    else:
        raise Exception("Multi Process Not Terminated!!")


def run_single_calculation(channel_list, tempFile_prefix, workDirectory, baseDirectory):
    result = []
    for i in range(len(channel_list)):
        temp_res = process_SingleImage_pool((channel_list[i], tempFile_prefix[i], workDirectory, baseDirectory))
        result.append(temp_res)
    return result


def calculate_birth_point(x,
                          workDirectory,
                          baseDirectory,
                          data_format = "channel_last",
                          UseParallel = True,
                          ):

    """
        Calculate the birth points of feature maps in a given layer.

        Parameters
        ----------
        x: 4-D array.
            The feature maps within a layer.

        workDirectory: path-like string.
            Path to the directory that generates the intermediate files. The default value is ".".
       
        baseDirectory: path-like string.
            Path to the code directory. The default value is current dir.

        data_format: "channel_first" or "channel_first".
            The data format of the tensor. 

        UseParallel: bool.
            Flag to use CPU parallel.

        Returns
        -------
        res_list: list.
            The list stores all the Betti curves, where each element is the Betti curves of each image.

        bp_arr: 2-D array.
            The array of birth points, where x-axis denotes image index and y-axis denotes channel index.

    """

    print("Calculate Birth Points...")
    print("This may take a while. Please wait...")

    assert isinstance(x, np.ndarray)
    assert x.ndim == 4
    x_symmetric = symmetrization(x, data_format)

    res_list = []

    if data_format == "channel_first":
        bp_arr =  np.zeros(shape=(x_symmetric.shape[0], x_symmetric.shape[1]))
        for img_idx in range(x_symmetric.shape[0]):
            featureMaps_perimage = x_symmetric[img_idx, :, :, :]
            channel_list = [featureMaps_perimage[channel_i, :, :] for channel_i in range(featureMaps_perimage.shape[1])]
            tempFile_prefix = ["%d_%d"%(img_idx, channel_i) for channel_i in range(featureMaps_perimage.shape[1])]
            if UseParallel:
                workers_available = len(os.sched_getaffinity(0))
                assert workers_available > 0
                res = run_multiprocess_calculation(channel_list, tempFile_prefix, workDirectory, baseDirectory, num_worker=workers_available)
            else:
                res = run_single_calculation(channel_list, tempFile_prefix, workDirectory, baseDirectory)

            res_list.append(res)
            for channel_i, item in enumerate(res):
                # print(item)
                bp_arr[img_idx, channel_i] = birth_point(item)


    elif data_format == "channel_last": 
        bp_arr =  np.zeros(shape=(x_symmetric.shape[0], x_symmetric.shape[-1]))
        # for img_idx in tqdm.tqdm(range(x_symmetric.shape[0]), desc="Calculation: "):
        for img_idx in range(x_symmetric.shape[0]):
            featureMaps_perimage = x_symmetric[img_idx, :, :, :]
            channel_list = [featureMaps_perimage[:,:,channel_i] for channel_i in range(featureMaps_perimage.shape[-1])]
            tempFile_prefix = ["%d_%d"%(img_idx, channel_i) for channel_i in range(featureMaps_perimage.shape[-1])]
            if UseParallel:
                workers_available = len(os.sched_getaffinity(0))
                assert workers_available > 0
                res = run_multiprocess_calculation(channel_list, tempFile_prefix, workDirectory, baseDirectory, num_worker=workers_available)
            else:
                res = run_single_calculation(channel_list, tempFile_prefix, workDirectory, baseDirectory)

            res_list.append(res)
            for channel_i, item in enumerate(res):
                # print(item)
                bp_arr[img_idx, channel_i] = birth_point(item)      

    else:
        raise Exception("Data format Error! Please use channel_first or channel_last.")

  

    return res_list, bp_arr


if __name__ == "__main__":
    pass