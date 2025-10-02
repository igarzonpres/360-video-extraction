# Feature Plan: Folder Filetree + Common Output (Updated)

This plan introduces a persistent file tree on the left side of the GUI to select a subfolder (per-video), moves the parent-folder picker into that pane with a functional Refresh, and adds a Settings option to select a Common Output Folder with a toggle to either merge results across videos or keep them separated by video. All processing remains per-video; the output routing changes how results are organized on disk.

## Goals
- Always-visible file tree to browse subfolders under a chosen parent folder.
- Selecting a subfolder focuses the app on the video inside that subfolder.
- Move the parent folder "Select Parent Folder…" to the bottom of the file tree panel and add a working "Refresh" beside it.
- Add Settings option for a Common Output Folder.
- Add a toggle to choose between:
  - Merge: write results from all videos into a single common output tree.
  - Separate: write results under the common output root but separated by video.
- Preserve existing functionality (preview, overlay, splitting, aligning) against the selected subfolder.

## Non-Goals
- Changing the core pipeline algorithms.
- Networked browsing or deep recursive listings (limit to one level of subfolders).
- Persisting settings to disk (optional; see Stretch Goals).

## UX/Behavior Details
- Left panel (always visible):
  - Header label (e.g., "Project Browser").
  - `ttk.Treeview` listing immediate subfolders of the selected parent folder that contain at least one valid video (`.mp4, .mov, .avi, .mkv`).
  - Empty state when no parent folder is set.
  - Bottom row: `[Select Parent Folder…] [Refresh]`.
    - Refresh rescans the filesystem and repopulates the tree; keeps the current selection if it still exists.
- Right panel: existing tabs remain. Selecting a subfolder updates the current project used by Preview/Overlay/Splitting/Align.
- Common Output Folder (Settings):
  - Adds a path picker.
  - Adds a mode toggle:
    - Merge (default): Results from all videos go into the same common output tree preserving the internal structure (e.g., `output/images/pano_camera0` holds frames from every video).
    - Separate: Results remain separated by video under the common root.

## Output Routing Semantics (Final)
- Merge mode (all videos share one output tree):
  - Frames: always written locally under the selected video folder (e.g., `<video-folder>/frames`).
  - Outputs (everything under `output/` such as `images/`, `colmap_masks_yolo/`, `masked_images/`, etc.): mirrored under the common root at `<COMMON_OUTPUT>/output/...` so all videos’ results co-exist. No filename changes.
  - Anything else (logs, preview, configs) remains local to the video folder.
- Separate-by-Video mode:
  - Everything is written locally in the video folder (unchanged behavior).
  - Additionally, the entire `output/` folder is mirrored to `<COMMON_OUTPUT>/<video-name>/output/`.
  - No filename changes.

## Technical Design
- UI layout:
  - Introduce `LeftPanel` (fixed width) and keep the existing content as `RightPanel`.
  - `LeftPanel` contains the Treeview and the bottom-row controls (Select Parent Folder + Refresh).
- State:
  - `parent_folder: Optional[Path]` — chosen parent folder.
  - `selected_subfolder: Optional[Path]` — the subfolder chosen in the tree; drives `_selected_project`.
  - `common_output_root: Optional[Path]` — selected in Settings; `None` when disabled.
  - `common_output_merge: bool` — true for Merge mode; false for Separate mode.
- Tree population:
  - On selecting parent folder or pressing Refresh, scan immediate child directories; include those with at least one valid video.
  - Item `iid` is the absolute subfolder path; label is folder name.
  - Bind `<<TreeviewSelect>>` to set `_selected_project` and refresh action buttons.
- Output path helpers:
  - Adds `mirror_outputs(project_root: Path)` helper to copy the entire `project_root/output` tree into the destination:
    - Merge: `<COMMON_OUTPUT>/output/...`
    - Separate: `<COMMON_OUTPUT>/<video-name>/output/...`
  - Keeps all native writes local; mirrors after key stages so external scripts remain unchanged.
  - No filename changes in any mode; input reads remain anchored to the selected subfolder.

## Touch Points (initial scan)
- `run_gui.py`:
  - Add `LeftPanel` and move the Browse (Select Parent Folder) into it; add Refresh.
  - Add Settings control for Common Output Folder and a toggle (Merge vs Separate).
  - Track `parent_folder`, `selected_subfolder`, `common_output_root`, `common_output_merge`.
  - Update `start_pipeline_with_path` and actions (`on_start_split`, `on_start_align`) to use `selected_subfolder`.
  - Add and use `get_output_base()` and `maybe_prefix_for_merge()` for all write paths and emitted filenames.
- Pipeline helpers writing to `output/` (in `run_gui.py`):
  - Keep original local writes.
  - Use `mirror_outputs()` to reflect final state into the configured common root according to the selected mode.

## Risks & Mitigations
- Collision risk when merging outputs into shared directories.
  - Mitigation: frames are already video-prefixed; no filename changes will be introduced. If a pipeline stage lacks video-prefixed names, this may overwrite files; we will document this constraint rather than modify filenames.
- UI layout disruption (pack/grid mixing).
  - Mitigation: keep changes localized; left panel uses `pack(side=LEFT)` and right panel uses existing layout.
- Performance with very large merged directories.
  - Mitigation: no functional change required; document that Merge mode can create large folders.

## Validation
- Manual:
  - Select a parent folder with several subfolders (each with a video).
  - Switch subfolders and confirm preview/overlay operate on the selected video.
  - Set a Common Output Folder; test both modes:
    - Merge: outputs land in `<COMMON_OUTPUT>/output/...` with per-video filename prefixes.
    - Separate: outputs land in `<COMMON_OUTPUT>/<subfolder_name>/output/...`.
  - Use Refresh to pick up new/deleted subfolders; verify selection persists when possible.
- Automated:
  - Unit tests for `get_output_base()` (no common, separate, merge).
  - Unit tests for tree filtering (include only subfolders with valid videos).

## Stretch Goals
- Persist `parent_folder`, `common_output_root`, and `common_output_merge` in a small JSON config and reload on startup.
- Show per-folder video count in the tree.

## TODO (Implementation Steps)
1. Scaffold LeftPanel Treeview and split layout.
2. Add bottom row: Select Parent Folder + functional Refresh.
3. Add Settings: Common Output Folder picker + Merge/Separate toggle.
4. Track `parent_folder`, populate/refresh tree with video subfolders.
5. Bind selection to set `_selected_project` and refresh UI/controls.
6. Implement `get_output_base(project_root)` helper.
7. Implement `mirror_outputs(project_root)` and invoke after key stages.
8. Add focused tests (path logic, tree filtering).
9. Smoke-test UI flows manually.
