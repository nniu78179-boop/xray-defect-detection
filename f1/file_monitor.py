import shutil
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "save"
INPUT_DIR = PROJECT_ROOT / "input"
F2_DIR = PROJECT_ROOT / "f2"
F2_COMMAND = ["cmd", "/c", "run_crop_yolo.bat"]
POLL_INTERVAL = 1.0


def print_header():
    print("=" * 60)
    print("  文件自动监控转发程序")
    print("=" * 60)
    print(f"  监控目录: {SAVE_DIR}")
    print(f"  转发目录: {INPUT_DIR}")
    print(f"  裁剪程序: {F2_DIR / 'run_crop_yolo.bat'}")
    print(f"  轮询间隔: {POLL_INTERVAL} 秒")
    print("-" * 60)


def ensure_dirs():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  已确认监控目录: {SAVE_DIR}")
    print(f"  已确认转发目录: {INPUT_DIR}")
    print("-" * 60)


def iter_files(base_dir: Path):
    for path in base_dir.rglob("*"):
        if path.is_file():
            yield path


def get_target_path(source_path: Path) -> Path:
    rel_path = source_path.relative_to(SAVE_DIR)
    return INPUT_DIR / rel_path


def is_file_ready(path: Path) -> bool:
    try:
        size_1 = path.stat().st_size
        time.sleep(0.2)
        size_2 = path.stat().st_size
        return size_1 == size_2
    except OSError:
        return False


def trigger_f2(process: subprocess.Popen | None) -> subprocess.Popen | None:
    if process is not None and process.poll() is None:
        return process

    print("  [处理] 检测到新文件，启动 f2 图像裁剪...")
    return subprocess.Popen(F2_COMMAND, cwd=F2_DIR)


def copy_to_input(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def monitor_loop():
    print("  开始监控新文件...")
    print("  按 Ctrl+C 停止程序")
    print("=" * 60)

    known_files = {str(path) for path in iter_files(SAVE_DIR)}
    f2_process: subprocess.Popen | None = None
    rerun_f2 = False

    while True:
        time.sleep(POLL_INTERVAL)
        try:
            if f2_process is not None and f2_process.poll() is not None:
                exit_code = f2_process.returncode
                print(f"  [处理完成] f2 已退出，返回码: {exit_code}")
                f2_process = None
                if rerun_f2:
                    rerun_f2 = False
                    f2_process = trigger_f2(f2_process)

            current_files = {str(path) for path in iter_files(SAVE_DIR)}
            new_files = sorted(Path(path_text) for path_text in (current_files - known_files))
            copied_any = False

            for src in new_files:
                if not is_file_ready(src):
                    continue

                dst = get_target_path(src)
                try:
                    copy_to_input(src, dst)
                    print(f"  [新增] {src.name} -> {dst.relative_to(INPUT_DIR)}")
                    copied_any = True
                except Exception as exc:
                    print(f"  [错误] {src.name}: {exc}")

            known_files = {str(path) for path in iter_files(SAVE_DIR)}

            if copied_any:
                if f2_process is None or f2_process.poll() is not None:
                    f2_process = trigger_f2(f2_process)
                else:
                    rerun_f2 = True
                    print("  [排队] f2 正在运行，当前批次会在结束后继续处理")
        except Exception as exc:
            print(f"  [监控错误] {exc}")


def main():
    print_header()
    ensure_dirs()
    monitor_loop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("  程序已停止")
        print("=" * 60)
