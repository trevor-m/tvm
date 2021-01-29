# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-outside-toplevel, unused-argument, invalid-name
""" Common utilities used by TensorFlow frontend """
from .. import op
from ..dataflow_pattern import (
    is_constant,
    is_op,
    rewrite,
    is_tuple,
    wildcard,
    DFPatternCallback,
)


def multiclass_nms_pattern(boxes, scores, idxs, iou_threshold, num_boxes, indices):
    """TensorFlow OD api splits NMS into 90 branches.
    """
    one = is_constant()
    zero = is_constant()

    # Equivelent PyTorch code from above snippet
    # offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    cast = is_op("cast")(idxs)
    mx = is_op("max")(boxes)
    add = is_op("add")(mx, one)
    mul = is_op("multiply")(cast, add)

    # The following doesn't appear in the above Relay snippet. It is required for dynamic
    # stride_slice handling
    cast_like = is_op("cast_like")(zero, is_constant())
    less = is_op("less")(is_constant(), cast_like)
    shape_of = is_op("shape_of")(mul)
    cast_like = is_op("cast_like")(shape_of, is_constant())
    add = is_op("add")(is_constant(), cast_like)
    where = is_op("where")(less, add, is_constant())
    shape_of = is_op("shape_of")(mul)
    cast = is_op("cast")(shape_of)

    # This corresponds to offsets[:, None], where offsets is the result of multiplication
    dyn_strided_slice = is_op("dyn.strided_slice")(mul, where, cast, is_constant())

    # Add offsets to the boxes
    expand_dims = is_op("expand_dims")(dyn_strided_slice)
    add = is_op("add")(boxes, expand_dims)

    # The rest of patterns correspond to the PyTorch frontend conversion
    # function for torchvision::nms
    score_expand_dims = is_op("expand_dims")(scores)
    tup = is_tuple([score_expand_dims, add])
    concat = is_op("concatenate")(tup)
    data = is_op("expand_dims")(concat)

    nms = is_op("vision.non_max_suppression")(
        data, num_boxes, indices, is_constant(), iou_threshold
    )



    nms_per_class_outputs = []
    for i in range(num_classes):
        boxes = wildcard() # %1656 /* ty=Tensor[(?, 4), float32] */;
        scores = wildcard() # %1658 /* ty=Tensor[(?), float32] */;
        scores = is_op("expand_dims")(scores)
        data = is_op("concatenate")(is_tuple([scores, boxes])) # %1661 (?, 5)
        data = is_op("expand_dims")(data) # %1662 (1, ?, 5)

        # %1663 = vision.get_valid_counts(%1662, 0f /* ty=float32 */, meta[relay.attrs.GetValidCountsAttrs][0]) /* ty=(Tensor[(1), int32], Tensor[(1, ?, 5), float32], Tensor[(1, ?), int32]) */;
        get_valid_counts = is_op("vision.get_valid_counts")(data, is_constant())
        data = is_tuple_get_item(get_valid_counts, 1)
        count = is_tuple_get_item(get_valid_counts, 0)
        indices = is_tuple_get_item(get_valid_counts, 2)

        # Max output size - could be wildcard?
        max_output_size = is_op("shape_of")(boxes)
        # TODO(trevmorr): non-dyn strided_slice constant? or attrs
        max_output_size = is_op("strided_slice")(max_output_size, is_constant(), is_constant(), is_constant())
        max_output_size = is_op("squeeze")(max_output_size)
        max_output_size = is_op("minimum")(is_constant(), max_output_size)

        iou_threshold = is_constant() # TODO(trevmorr): wildcard?
        nms = is_op("vision.non_max_suppression")(
            data, count, indices, max_output_size, iou_threshold
        )
        nms_indices = is_tuple_get_item(nms, 0)
        nms_indices = is_op("squeeze")(nms_indices)

        # TODO(trevmorr): slice_begin might be able to be wildcard
        slice_begin = is_op("shape_of")(nms_indices)
        slice_begin = is_op("cast_like")(slice_begin, is_constant())
        slice_begin = is_op("add")(is_constant(), slice_begin)
        slice_begin = is_op("where")(is_constant(), slice_begin, is_constant())

        nms_count = is_tuple_get_item(pattern, 1)
        nms_count = is_op("squeeze")(nms_count)

        nms_indices = is_op("dyn.strided_slice")(nms_indices, slice_begin, nms_count, is_constant())

        nmsed_boxes = is_op("take")(boxes, nms_indices) # %1681 = take(%1656, %1680, axis=0) /* ty=Tensor[(?, 4), float32] */;
        nmsed_scores = is_op("take")(scores, nms_indices) # %7825 = take(%1658, %1680, axis=0) /* ty=Tensor[(?), float32] */;
        nms_per_class_outputs.append(nmsed_boxes)
    tup = is_tuple(nms_per_class_outputs)
    return is_op("concatenate")(tup)


class NMSRewrite(DFPatternCallback):
    """A callback to rewrite nms and restore batched nms"""

    def __init__(self):
        super().__init__()
        # exprs to extract
        self.boxes = wildcard()
        self.scores = wildcard()
        self.idxs = wildcard()
        self.iou_threshold = wildcard()
        self.num_boxes = wildcard()
        self.indices = wildcard()

        self.pattern = batched_nms_pattern(
            self.boxes,
            self.scores,
            self.idxs,
            self.iou_threshold,
            self.num_boxes,
            self.indices,
        )

    def convert_multiclass_nms(self, boxes, scores, idxs, iou_thres, num_boxes, indices):
        """Restore class-aware NMS using extracted class indices"""
        scores = op.expand_dims(scores, axis=-1, num_newaxis=1)
        idxs = op.expand_dims(idxs, axis=-1, num_newaxis=1)
        idxs = op.cast(idxs, "float32")
        data = op.concatenate([idxs, scores, boxes], -1)
        data = op.expand_dims(data, 0, 1)

        top_k = max_out_size = -1
        out = op.vision.non_max_suppression(
            data=data,
            valid_count=num_boxes,
            indices=indices,
            max_output_size=max_out_size,
            iou_threshold=iou_thres,
            force_suppress=False,
            top_k=top_k,
            coord_start=2,
            score_index=1,
            id_index=0,
            return_indices=True,
            invalid_to_bottom=False,
        )
        return out.tuple_value

    def callback(self, pre, post, node_map):
        boxes = node_map[self.boxes][0]
        scores = node_map[self.scores][0]
        idxs = node_map[self.idxs][0]
        iou_thres = node_map[self.iou_threshold][0]
        num_boxes = node_map[self.num_boxes][0]
        indices = node_map[self.indices][0]
        return self.convert_batched_nms(boxes, scores, idxs, iou_thres, num_boxes, indices)


def rewrite_nms_to_multiclass_nms(mod):
    """Rewrite the input graph to replace non maximum surpression
    in torchvision that does not take class id into account with the one
    that avoids IOU tests between different classes.
    """
    mod["main"] = rewrite(NMSRewrite(), mod["main"])
    return mod