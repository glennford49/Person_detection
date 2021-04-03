import sys
import os
import cv2
import time
import numpy as np
from openvino.inference_engine import IECore
model ="model/pedestrian-detection-adas-0002.xml"
device ="CPU"
input_stream = "video/person_media.mp4"
def main():
    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))
    assert len(net.outputs) == 1, "Demo supports only single output topologies"
    out_blob = next(iter(net.outputs))
    exec_net = ie.load_network(network=net, num_requests=2, device_name=device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + input_stream
    cur_request_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_h, frame_w = frame.shape[:2]
        inf_start = time.time()
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))
        
        feed_dict[input_blob] = in_frame
        exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            # Draw performance stats
            inf_time_message = "Fps: {:.2f}".format(det_time * 1000) 
            cv2.rectangle(frame, (4,3),(95,20),(0,0,0),-1)
            cv2.putText(frame, inf_time_message, (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255,255), 1)
        cv2.imshow("Detections", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    sys.exit(main() or 0)
