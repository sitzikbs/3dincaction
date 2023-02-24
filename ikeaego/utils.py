# utils for ikea ego action understanding
import numpy as np
import json


def find_label_segments_and_scores(input, mode="logits"):
    """
    finds the range of label segments in a given labels array and computes the segment score as max logit average

    Parameters
    ----------
    logits : numpy array of per frame logits
    mode : logits | labels
    Returns
    -------
    ranges : list of segments ranges
    scores : segment score (logits average)
    segment_label : the label of the corresponding segment
    """
    if mode == "logits":
        logits = input
        labels = np.argmax(logits, axis=1)
    else:
        labels = input

    diff = np.diff(labels)
    iszero = np.concatenate(([0], np.equal(diff, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    segments = np.where(absdiff == 1)[0].reshape(-1, 2)
    scores = []
    segment_label = []
    for segment in segments:
        segment_label.append(labels[segment[0]])
        if mode == "logits":
            scores.append(np.mean(logits[segment[0]:segment[1], labels[segment[0]]]))
        else:
            scores.append(1.0)
    return segments, np.array(scores), segment_label


def convert_frame_logits_to_segment_json(logits, json_filename, video_name_list, action_list, mode="logits",
                                         details=""):
    """
    convert dataset per frame logits to scored segments and save to .json file in the ActivityNet format
    for action localization evaluation
    http://activity-net.org/challenges/2020/tasks/anet_localization.html

    Parameters
    ----------
    logits : per frame logits or labels (depending on mode)
    json_filename : output .json file name (full path)
    video_name_list : list of video names
    action_list : list of action labels
    mode : logits | labels
     Returns
    -------
    """
    json_dict_to_write = {"version": "VERSION 1.3"}

    results = {}
    for i, vid_logits in enumerate(logits):
        segments, scores, segment_labels = find_label_segments_and_scores(vid_logits, mode=mode)
        list_of_result_dicts = []
        vid_name = video_name_list[i]
        for j, segment in enumerate(segments):
            list_of_result_dicts.append({"label": action_list[segment_labels[j]],
                                         "score": float(scores[j]),
                                         "segment": segment.tolist()})
        results[vid_name] = list_of_result_dicts
    json_dict_to_write["results"] = results
    json_dict_to_write["external_data"] = {"details": details}
    print("Saving file: " + str(json_filename))
    with open(json_filename, 'w') as outfile:
        json.dump(json_dict_to_write, outfile)


def accume_per_video_predictions(vid_idx, frame_pad, pred_labels_per_video, logits_per_video, pred_labels,
                                 logits, frames_per_clip):
    """
    This is a helper function to accumulate the predictions of the different batches into a single list
    containing the predictions for each video separately. It is used in all of the test files except the frame based
    (no sequence)
    Parameters
    ----------
    vid_idx : list of video index corresponding for each element in the batch
    frame_pad : list of number of padded frames per element in the batch
    pred_labels_per_video : predicted labels per video - accumulated from previous batch
    logits_per_video : logits per video - accumulated from previous batch
    pred_labels : the current batch predictions
    logits : the current batch logits
    frames_per_clip : number of frames per clip (int)

    Returns
    -------
        pred_labels_per_video : predicted labels per video - accumulated from previous batch
        logits_per_video : logits per video - accumulated from previous batch
    """

    for i in range(len(vid_idx)):
        batch_vid_idx = vid_idx[i].item()
        batch_frame_pad = frame_pad[i].item()

        pred_labels_per_video[batch_vid_idx].extend(pred_labels[i*frames_per_clip:(i+1)*frames_per_clip])
        if not batch_frame_pad == 0:
            pred_labels_per_video[batch_vid_idx] = pred_labels_per_video[batch_vid_idx][0:-batch_frame_pad]


        logits_per_video[batch_vid_idx].extend(logits[i*frames_per_clip:(i+1)*frames_per_clip])
        if not batch_frame_pad == 0:
            logits_per_video[batch_vid_idx] = logits_per_video[batch_vid_idx][0:-batch_frame_pad]

    return pred_labels_per_video, logits_per_video

