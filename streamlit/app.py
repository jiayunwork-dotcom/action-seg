import os
import time
import tempfile
from io import StringIO
from typing import Optional

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import cv2
from PIL import Image

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")


def api_request(method, endpoint, **kwargs):
    url = f"{API_BASE}{endpoint}"
    try:
        response = requests.request(method, url, timeout=300, **kwargs)
        return response
    except requests.exceptions.RequestException as e:
        return None


def load_results(video_id, model_version="latest"):
    results_resp = api_request(
        "GET", f"/videos/{video_id}/results", params={"model_version": model_version}
    )
    if results_resp and results_resp.status_code == 200:
        return results_resp.json()
    return None


def load_timeline(video_id, model_version="latest"):
    timeline_resp = api_request(
        "GET", f"/videos/{video_id}/results/timeline", params={"model_version": model_version}
    )
    if timeline_resp and timeline_resp.status_code == 200:
        return timeline_resp.json()
    return None


def update_segment(video_id, seg_index, start_time=None, end_time=None, action_id=None, model_version="latest"):
    data = {}
    if start_time is not None:
        data["start_time"] = start_time
    if end_time is not None:
        data["end_time"] = end_time
    if action_id is not None:
        data["action_id"] = action_id
    resp = api_request(
        "PUT",
        f"/videos/{video_id}/segments/{seg_index}",
        json=data,
        params={"model_version": model_version},
    )
    if resp and resp.status_code == 200:
        return resp.json()
    return None


def split_segment(video_id, seg_index, split_time, model_version="latest"):
    resp = api_request(
        "POST",
        f"/videos/{video_id}/segments/{seg_index}/split",
        json={"split_time": split_time},
        params={"model_version": model_version},
    )
    if resp and resp.status_code == 200:
        return resp.json()
    return None


def merge_segments(video_id, idx1, idx2, model_version="latest"):
    resp = api_request(
        "POST",
        f"/videos/{video_id}/segments/merge",
        json={"segment_index_1": idx1, "segment_index_2": idx2},
        params={"model_version": model_version},
    )
    if resp and resp.status_code == 200:
        return resp.json()
    return None


def undo_edit(video_id, model_version="latest"):
    resp = api_request(
        "POST",
        f"/videos/{video_id}/segments/undo",
        params={"model_version": model_version},
    )
    if resp and resp.status_code == 200:
        return resp.json()
    return None


def can_undo(video_id, model_version="latest"):
    resp = api_request(
        "GET",
        f"/videos/{video_id}/segments/can-undo",
        params={"model_version": model_version},
    )
    if resp and resp.status_code == 200:
        return resp.json().get("can_undo", False)
    return False


def export_file(video_id, format_type, model_version="latest"):
    resp = api_request(
        "GET",
        f"/videos/{video_id}/export/{format_type}",
        params={"model_version": model_version},
    )
    if resp and resp.status_code == 200:
        filename = resp.headers.get("Content-Disposition", "").split('filename="')[-1].rstrip('"')
        return resp.content, filename
    return None, None


def render_timeline(segments, action_classes_results, video_duration):
    fig_timeline = go.Figure()

    for i, seg in enumerate(segments):
        seg_duration = seg["end_time"] - seg["start_time"]
        if seg_duration <= 0:
            continue
        color = next(
            (c["color"] for c in action_classes_results if c["id"] == seg["action_id"]),
            "#808080"
        )
        fig_timeline.add_trace(go.Bar(
            x=[seg_duration],
            y=[f"#{i+1} {seg['action_name']}"],
            orientation="h",
            base=[seg["start_time"]],
            marker_color=color,
            name=f"{seg['action_name']} ({seg['start_time']:.1f}s-{seg['end_time']:.1f}s)",
            hovertext=(
                f"片段 #{i+1}<br>"
                f"动作: {seg['action_name']}<br>"
                f"时间: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s<br>"
                f"置信度: {seg['confidence']:.2%}<br>"
                f"帧: {seg['start_frame']} - {seg['end_frame']}"
            ),
            hoverinfo="text",
        ))

    fig_timeline.update_layout(
        barmode="overlay",
        height=400,
        xaxis_title="时间 (秒)",
        xaxis_range=[0, video_duration],
        yaxis_title="动作片段",
        showlegend=True,
    )
    st.plotly_chart(fig_timeline, use_container_width=True)


st.set_page_config(
    page_title="视频动作识别与时序行为分割",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 视频动作识别与时序行为分割系统")
st.markdown("---")

st.sidebar.title("系统信息")
st.sidebar.info("对未修剪的长视频进行逐帧动作标注和动作片段定位")

if "analysis_error" not in st.session_state:
    st.session_state["analysis_error"] = None
if "analysis_status" not in st.session_state:
    st.session_state["analysis_status"] = "idle"

tab1, tab2, tab3 = st.tabs(["📤 上传与分析", "📊 结果展示", "📈 评估对比"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("视频输入")
        input_mode = st.radio("选择输入方式", ["上传视频文件", "视频URL"])

        video_file = None
        video_url = None

        if input_mode == "上传视频文件":
            video_file = st.file_uploader(
                "选择视频文件 (MP4/AVI/MOV, 最大500MB)",
                type=["mp4", "avi", "mov"],
                accept_multiple_files=False,
            )
        else:
            video_url = st.text_input("输入视频URL", placeholder="https://example.com/video.mp4")

        model_versions_resp = api_request("GET", "/models/versions")
        available_versions = ["latest"]
        action_classes = []
        if model_versions_resp and model_versions_resp.status_code == 200:
            mv_data = model_versions_resp.json()
            available_versions = mv_data.get("versions", ["latest"])
            action_classes = mv_data.get("action_classes", [])

        selected_model = st.selectbox("选择模型版本", available_versions, index=0)

        analyze_btn = st.button("🚀 开始分析", type="primary", disabled=False)

    with col2:
        st.subheader("任务进度")
        progress_bar = st.progress(0)
        status_text = st.empty()
        task_id_placeholder = st.empty()

        if st.session_state.get("analysis_error"):
            st.error(f"❌ {st.session_state['analysis_error']}")
            if st.button("🔄 重试分析", key="retry_btn"):
                st.session_state["analysis_error"] = None
                st.session_state["analysis_status"] = "idle"
                st.rerun()

        if st.session_state.get("analysis_status") == "success":
            st.success("✅ 上次分析已完成")
            if st.button("🔄 重新分析", key="reanalyze_btn"):
                st.session_state["analysis_status"] = "idle"
                st.rerun()

    video_id = None
    current_task_id = None

    if video_file is not None or video_url is not None:
        if analyze_btn:
            st.session_state["analysis_error"] = None
            st.session_state["analysis_status"] = "running"

            try:
                with st.spinner("上传视频中..."):
                    if video_file is not None:
                        files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
                        resp = api_request("POST", "/videos/upload", files=files)
                    else:
                        resp = api_request("POST", "/videos/upload-url", json={"url": video_url})

                if resp is None:
                    st.session_state["analysis_error"] = "API连接失败，请检查后端服务是否运行"
                    st.session_state["analysis_status"] = "error"
                    st.rerun()

                if resp.status_code != 200:
                    st.session_state["analysis_error"] = f"上传失败: {resp.text}"
                    st.session_state["analysis_status"] = "error"
                    st.rerun()

                upload_data = resp.json()
                video_id = upload_data["video_id"]
                st.success(f"✅ 视频上传成功! ID: {video_id}")

                with st.spinner("提交分析任务..."):
                    analyze_resp = api_request(
                        "POST",
                        f"/videos/{video_id}/analyze",
                        json={"model_version": selected_model},
                    )

                if analyze_resp is None:
                    st.session_state["analysis_error"] = "提交分析任务时API连接失败"
                    st.session_state["analysis_status"] = "error"
                    st.rerun()

                if analyze_resp.status_code != 200:
                    st.session_state["analysis_error"] = f"提交分析任务失败: {analyze_resp.text}"
                    st.session_state["analysis_status"] = "error"
                    st.rerun()

                analyze_data = analyze_resp.json()
                current_task_id = analyze_data["task_id"]
                task_id_placeholder.info(f"任务ID: {current_task_id}")
                st.session_state["current_task_id"] = current_task_id
                st.session_state["current_video_id"] = video_id

                with st.spinner("分析进行中..."):
                    max_polls = 600
                    poll_count = 0
                    while poll_count < max_polls:
                        try:
                            status_resp = api_request(
                                "GET", f"/tasks/{current_task_id}/status"
                            )

                            if status_resp is None:
                                status_text.warning("获取状态失败，2秒后重试...")
                                time.sleep(2)
                                poll_count += 1
                                continue

                            if status_resp.status_code != 200:
                                status_text.warning(f"状态查询异常({status_resp.status_code})，2秒后重试...")
                                time.sleep(2)
                                poll_count += 1
                                continue

                            status_data = status_resp.json()
                            progress = max(0, min(100, status_data["progress"]))
                            progress_bar.progress(progress / 100.0)
                            status_text.info(f"{status_data['message']} ({progress}%)")

                            if status_data["status"] == "SUCCESS":
                                progress_bar.progress(1.0)
                                status_text.success("✅ 分析完成!")
                                st.session_state["analysis_status"] = "success"
                                st.session_state["analysis_error"] = None
                                break
                            elif status_data["status"] == "FAILURE":
                                error_msg = status_data.get("error") or status_data.get("message") or "未知错误"
                                st.session_state["analysis_error"] = f"分析失败: {error_msg}"
                                st.session_state["analysis_status"] = "error"
                                progress_bar.empty()
                                status_text.error(f"❌ 分析失败: {error_msg}")
                                break
                            elif status_data["status"] in ("PENDING", "PROGRESS", "RETRY", "RECEIVED"):
                                pass
                            else:
                                status_text.warning(f"未知状态: {status_data['status']}")

                        except Exception as e:
                            status_text.warning(f"轮询异常: {str(e)}，2秒后重试...")

                        time.sleep(2)
                        poll_count += 1
                    else:
                        st.session_state["analysis_error"] = "分析超时，请稍后查看任务状态"
                        st.session_state["analysis_status"] = "error"
                        status_text.error("⏰ 轮询超时")

            except Exception as e:
                st.session_state["analysis_error"] = f"发生异常: {str(e)}"
                st.session_state["analysis_status"] = "error"
                st.rerun()

    if "current_video_id" in st.session_state:
        st.info(f"当前视频ID: {st.session_state['current_video_id']}")

with tab2:
    if "current_video_id" not in st.session_state:
        st.warning("请先在'上传与分析'标签页上传并分析视频")
    else:
        video_id = st.session_state["current_video_id"]
        model_version = selected_model if "selected_model" in locals() else "latest"
        segments = []
        video_info = {}
        action_classes_results = []
        timeline_data = None
        results_data = None
        video_duration = 0.0

        try:
            with st.spinner("加载分析结果..."):
                results_data = load_results(video_id, model_version)
                timeline_resp = api_request(
                    "GET", f"/videos/{video_id}/results/timeline", params={"model_version": model_version}
                )
                if timeline_resp and timeline_resp.status_code == 200:
                    timeline_data = timeline_resp.json()

            if results_data is None:
                st.error("❌ 无法连接API服务，请检查后端是否运行")
            elif not results_data.get("segments"):
                st.warning("分析结果尚未就绪，请先完成视频分析")
            else:
                segments = results_data["segments"]
                video_info = results_data["video_info"]
                action_classes_results = results_data["action_classes"]
                video_duration = video_info["duration"]

                st.subheader("⚙️ 编辑工具栏")
                col_tool1, col_tool2, col_tool3, col_tool4 = st.columns([1, 1, 1, 2])

                with col_tool1:
                    can_undo_flag = can_undo(video_id, model_version)
                    if st.button("↩️ 撤销", disabled=not can_undo_flag, type="secondary"):
                        undo_result = undo_edit(video_id, model_version)
                        if undo_result and undo_result.get("success"):
                            st.success("✅ 撤销成功")
                            st.rerun()
                        else:
                            st.warning("无法撤销")

                with col_tool2:
                    edit_mode = st.toggle("✏️ 编辑模式", value=False)

                with col_tool3:
                    with st.expander("📤 导出"):
                        export_format = st.radio(
                            "选择导出格式",
                            ["JSON", "SRT字幕", "CSV表格"],
                            horizontal=True,
                        )
                        format_map = {"JSON": "json", "SRT字幕": "srt", "CSV表格": "csv"}
                        if st.button("导出文件", type="primary", key="export_btn"):
                            fmt = format_map[export_format]
                            content, filename = export_file(video_id, fmt, model_version)
                            if content:
                                st.success("✅ 导出成功，请点击下方按钮下载")
                                st.download_button(
                                    label="💾 下载文件",
                                    data=content,
                                    file_name=filename,
                                    mime="application/octet-stream",
                                    key="download_export",
                                )
                            else:
                                st.error("导出失败")

                with col_tool4:
                    st.info(f"共 {len(segments)} 个动作片段 | 视频时长: {video_duration:.2f}s")

                col_vid, col_info = st.columns([3, 1])

                with col_vid:
                    st.subheader("🎥 视频播放")
                    if video_file is not None and "temp_video_path" not in st.session_state:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(video_file.getvalue())
                        tfile.close()
                        st.session_state["temp_video_path"] = tfile.name

                    if "temp_video_path" in st.session_state:
                        st.video(st.session_state["temp_video_path"])
                    else:
                        st.info("视频播放器: 请在左侧上传视频以播放")

                    if segments and timeline_data:
                        st.subheader("⏱️ 动作时间轴")
                        render_timeline(segments, action_classes_results, video_duration)

                with col_info:
                    st.subheader("📋 视频信息")
                    st.json({
                        "时长": f"{video_info['duration']:.2f}s",
                        "原始帧率": f"{video_info['fps']:.1f} fps",
                        "采样帧率": f"{video_info['target_fps']} fps",
                        "分辨率": f"{video_info['width']}x{video_info['height']}",
                        "总帧数": video_info["total_frames"],
                        "采样帧数": video_info["sample_frames_count"],
                    })

                    st.subheader("📊 类别分布")
                    if segments:
                        seg_df = pd.DataFrame(segments)
                        seg_df = seg_df[seg_df["end_time"] > seg_df["start_time"]]
                        class_durations = seg_df.groupby("action_name").apply(
                            lambda x: (x["end_time"] - x["start_time"]).sum()
                        ).reset_index(name="duration")
                        class_durations = class_durations.sort_values("duration", ascending=False)

                        color_map = {c["name"]: c["color"] for c in action_classes_results}
                        fig_pie = px.pie(
                            class_durations,
                            values="duration",
                            names="action_name",
                            color="action_name",
                            color_discrete_map=color_map,
                            title="动作时长占比",
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("---")
                st.subheader("🎯 动作片段列表")

                if edit_mode:
                    st.info("💡 编辑模式：展开每个片段可编辑属性、拆分；上方可合并相邻片段")

                    merge_col1, merge_col2, merge_col3 = st.columns([1, 1, 1])
                    with merge_col1:
                        max_idx = max(0, len(segments) - 1)
                        merge_idx1 = st.number_input(
                            "合并 - 片段1索引",
                            min_value=0,
                            max_value=max_idx,
                            value=0,
                            step=1,
                            key="merge_idx1",
                        )
                    with merge_col2:
                        merge_idx2 = st.number_input(
                            "合并 - 片段2索引",
                            min_value=0,
                            max_value=max_idx,
                            value=min(1, max_idx),
                            step=1,
                            key="merge_idx2",
                        )
                    with merge_col3:
                        if st.button("🔗 合并选中片段", type="primary", key="merge_btn"):
                            if abs(merge_idx1 - merge_idx2) == 1:
                                result = merge_segments(video_id, merge_idx1, merge_idx2, model_version)
                                if result:
                                    st.success("✅ 合并成功")
                                    st.rerun()
                                else:
                                    st.error("❌ 合并失败")
                            else:
                                st.error("只能合并相邻的两个片段")

                    for i, seg in enumerate(segments):
                        with st.expander(f"📌 片段 #{i+1}: {seg['action_name']} ({seg['start_time']:.2f}s - {seg['end_time']:.2f}s)"):
                            col_e1, col_e2 = st.columns([2, 1])

                            with col_e1:
                                new_start = st.number_input(
                                    "起始时间 (秒)",
                                    min_value=0.0,
                                    max_value=float(video_duration),
                                    value=float(seg["start_time"]),
                                    step=0.1,
                                    key=f"start_{i}",
                                )
                                new_end = st.number_input(
                                    "结束时间 (秒)",
                                    min_value=0.0,
                                    max_value=float(video_duration),
                                    value=float(seg["end_time"]),
                                    step=0.1,
                                    key=f"end_{i}",
                                )
                                action_names = [c["name"] for c in action_classes_results]
                                action_ids = [c["id"] for c in action_classes_results]
                                current_name_idx = action_ids.index(seg["action_id"]) if seg["action_id"] in action_ids else 0
                                new_action_name = st.selectbox(
                                    "动作类别",
                                    action_names,
                                    index=current_name_idx,
                                    key=f"action_{i}",
                                )
                                new_action_id = action_ids[action_names.index(new_action_name)]

                            with col_e2:
                                st.metric("持续时间", f"{seg['end_time'] - seg['start_time']:.2f}s")
                                st.metric("置信度", f"{seg['confidence']:.2%}")

                                changed = (
                                    abs(new_start - seg["start_time"]) > 0.001
                                    or abs(new_end - seg["end_time"]) > 0.001
                                    or new_action_id != seg["action_id"]
                                )

                                if st.button("💾 保存修改", key=f"save_{i}", disabled=not changed, type="primary"):
                                    result = update_segment(
                                        video_id, i,
                                        start_time=new_start,
                                        end_time=new_end,
                                        action_id=new_action_id,
                                        model_version=model_version,
                                    )
                                    if result:
                                        st.success("✅ 修改已保存")
                                        st.rerun()
                                    else:
                                        st.error("❌ 保存失败")

                                st.divider()

                                split_time_val = st.number_input(
                                    "拆分时间点 (秒)",
                                    min_value=float(seg["start_time"]),
                                    max_value=float(seg["end_time"]),
                                    value=float((seg["start_time"] + seg["end_time"]) / 2),
                                    step=0.1,
                                    key=f"split_time_{i}",
                                )
                                if st.button("✂️ 拆分段", key=f"split_{i}"):
                                    result = split_segment(video_id, i, split_time_val, model_version)
                                    if result:
                                        st.success("✅ 拆分成功")
                                        st.rerun()
                                    else:
                                        st.error("❌ 拆分失败")

                else:
                    if segments:
                        seg_display_df = pd.DataFrame(segments)
                        seg_display_df = seg_display_df[[
                            "action_name", "start_time", "end_time",
                            "start_frame", "end_frame", "confidence"
                        ]]
                        seg_display_df.columns = [
                            "动作类别", "开始时间(s)", "结束时间(s)",
                            "开始帧", "结束帧", "置信度"
                        ]
                        seg_display_df["置信度"] = seg_display_df["置信度"].apply(lambda x: f"{x:.2%}")
                        seg_display_df.insert(0, "序号", range(1, len(seg_display_df) + 1))

                        st.dataframe(
                            seg_display_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                if results_data.get("frame_predictions"):
                    st.subheader("📈 置信度时序折线图")
                    probs = np.array(results_data["frame_predictions"]["probabilities"])
                    labels = np.array(results_data["frame_predictions"]["labels"])
                    fps = video_info["target_fps"]
                    times = np.arange(len(labels)) / fps

                    id_to_name = {c["id"]: c["name"] for c in action_classes_results}
                    id_to_color = {c["id"]: c["color"] for c in action_classes_results}

                    top_confidences = np.max(probs, axis=1)

                    fig_conf = go.Figure()
                    fig_conf.add_trace(go.Scatter(
                        x=times,
                        y=top_confidences,
                        mode="lines",
                        name="最大置信度",
                        line=dict(color="#1f77b4", width=1),
                    ))

                    bg_id = 0
                    for cid, cname in id_to_name.items():
                        if cid == bg_id:
                            continue
                        mask = labels == cid
                        if np.any(mask):
                            fig_conf.add_trace(go.Scatter(
                                x=times[mask],
                                y=top_confidences[mask],
                                mode="markers",
                                name=cname,
                                marker=dict(color=id_to_color.get(cid, "#808080"), size=4),
                            ))

                    fig_conf.update_layout(
                        xaxis_title="时间 (秒)",
                        yaxis_title="置信度",
                        yaxis_range=[0, 1],
                        height=400,
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 加载结果时出错: {str(e)}")
            if st.button("🔄 重试加载"):
                st.rerun()

with tab3:
    st.subheader("📈 评估指标对比")

    gt_file = st.file_uploader(
        "上传Ground Truth标注文件 (CSV格式: start_frame,end_frame,action_label)",
        type=["csv"],
    )

    if "current_video_id" in st.session_state and gt_file is not None:
        if st.button("开始评估"):
            try:
                video_id = st.session_state["current_video_id"]
                files = {"gt_file": (gt_file.name, gt_file.getvalue(), "text/csv")}
                eval_resp = api_request(
                    "POST",
                    f"/videos/{video_id}/evaluate",
                    files=files,
                    params={"model_version": selected_model},
                )

                if eval_resp is None:
                    st.error("❌ API连接失败，请检查后端服务是否运行")
                elif eval_resp.status_code != 200:
                    st.error(f"❌ 评估失败: {eval_resp.text}")
                else:
                    eval_data = eval_resp.json()
                    metrics = eval_data["metrics"]

                    st.success("✅ 评估完成!")

                    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                    with col_m1:
                        st.metric("逐帧准确率", f"{metrics['frame_accuracy']:.2%}")
                    with col_m2:
                        st.metric("编辑距离分数", f"{metrics['edit_score']:.2%}")
                    with col_m3:
                        st.metric("F1@10", f"{metrics['f1_at_10']:.2%}")
                    with col_m4:
                        st.metric("F1@25", f"{metrics['f1_at_25']:.2%}")
                    with col_m5:
                        st.metric("F1@50", f"{metrics['f1_at_50']:.2%}")

                    st.markdown("---")

                    metrics_names = [
                        "逐帧准确率", "编辑距离分数",
                        "F1@IoU=0.10", "F1@IoU=0.25", "F1@IoU=0.50"
                    ]
                    metrics_values = [
                        metrics["frame_accuracy"],
                        metrics["edit_score"],
                        metrics["f1_at_10"],
                        metrics["f1_at_25"],
                        metrics["f1_at_50"],
                    ]

                    fig_bar = go.Figure([go.Bar(
                        x=metrics_names,
                        y=metrics_values,
                        marker_color=[
                            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
                        ],
                        text=[f"{v:.2%}" for v in metrics_values],
                        textposition="auto",
                    )])
                    fig_bar.update_layout(
                        yaxis_title="分数",
                        yaxis_range=[0, 1],
                        height=500,
                        title="评估指标对比",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.subheader("📋 评估结果详情")
                    eval_df = pd.DataFrame({
                        "指标": metrics_names,
                        "数值": metrics_values,
                        "百分比": [f"{v:.2%}" for v in metrics_values],
                    })
                    st.table(eval_df)
            except Exception as e:
                st.error(f"❌ 评估过程出错: {str(e)}")
                if st.button("🔄 重试评估"):
                    st.rerun()
    else:
        st.info("请先上传分析视频，并提供Ground Truth标注CSV文件进行评估")
