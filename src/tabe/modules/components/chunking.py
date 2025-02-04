from typing import List

from src.tabe.utils.occlusion_utils import OcclusionLevel


def basic_video_chunker(occlusion_list: List[OcclusionLevel], min_frames: int = 14, ideal_frames: int = 30) -> List:
    assert occlusion_list[0] == OcclusionLevel.NO_OCCLUSION
    chunked_frames = []
    occl_found = False
    clip_inds = []
    for i, occl_info in enumerate(occlusion_list):
        if occl_info != OcclusionLevel.NO_OCCLUSION:
            occl_found = True
        else:
            if occl_found or len(clip_inds) >= ideal_frames:
                # End the current clip and get the next one going
                clip_inds.append(i)
                chunked_frames.append(clip_inds)
                clip_inds = []
                occl_found = False
            elif not occl_found and len(clip_inds) == min_frames:
                chunked_frames.append(clip_inds)
                clip_inds = []
        clip_inds.append(i)

    chunked_frames.append(clip_inds)

    return chunked_frames


def _combine_small_chunks(chunks: list, ideal_frames: int = 30):
    chunk_lengths = [len(chunk) for chunk in chunks]

    # Resulting list of combined chunks (with actual frames)
    combined_chunks = []
    current_chunk = []
    current_length = 0

    # Iterate through each frame and its length
    for frame, length in zip(chunks, chunk_lengths):
        # Check if adding this frame would keep the total within 30
        if current_length + length <= ideal_frames + 1:
            # Add frame to current chunk if within the limit
            current_chunk.append(frame)
            current_length += length
        else:
            # Otherwise, save the current chunk and start a new one
            if current_chunk:
                combined_chunks.append([item for sublist in current_chunk for item in sublist])
            current_chunk = [frame]
            current_length = length

    # Add any remaining frames in the last chunk
    if current_chunk:
        combined_chunks.append([item for sublist in current_chunk for item in sublist])

    return combined_chunks


def chunk_postprocess(chunks: list, occlusion_info: list[OcclusionLevel], ideal_frames: int = 30):
    """"
        Pad with extra non occluded
        Combine any that are too short
    """
    new_chunks = []
    for chunk in chunks:
        extra_before = []
        extra_after = []
        if len(chunk) < ideal_frames:
            poss_missing_frames = ideal_frames - len(chunk)
            for i in range(1, (poss_missing_frames // 2) + 1):
                if chunk[0] - i > 0 and occlusion_info[chunk[0] - i] == OcclusionLevel.NO_OCCLUSION:
                    extra_before.append(chunk[0] - i)
                else:
                    break
            for i in range(1, (poss_missing_frames // 2) + 1):
                if chunk[-1] + i < len(occlusion_info) and occlusion_info[chunk[-1] + i] == OcclusionLevel.NO_OCCLUSION:
                    extra_after.append(chunk[-1] + i)
                else:
                    break
        new_chunks.append(sorted(extra_before + chunk + extra_after))

    new_chunks = _combine_small_chunks(new_chunks, ideal_frames=ideal_frames)
    new_chunks = [sorted(list(set(chunk))) for chunk in new_chunks]

    return new_chunks
