import time
from pathlib import Path
import yt_dlp


class YouTubeDownloader:

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_downloaded_ids(self) -> set[str]:
        """Returns video IDs that already exist on disk (any subdir)."""
        return {p.stem for p in self.base_dir.rglob("*.mp4")}

    def download(self, url, jump_type, rotation):

        output_dir = self.base_dir / jump_type / rotation
        output_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 60,
            "sleep_interval": 1,
            "max_sleep_interval": 3,
            "nocheckcertificate": True,
            "http_headers": {"User-Agent": "Mozilla/5.0"},
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

            if not info:
                return None

            filepath = output_dir / f"{info['id']}.mp4"

            metadata = {
                "video_id": info["id"],
                "title": info.get("title"),
                "video_url": info.get("webpage_url"),
                "filepath": str(filepath),
                "jump_type": jump_type,
                "rotation": rotation,
            }

            return metadata

        except Exception as e:
            print("Download error:", e)
            return None


def extract_video_id(url):

    if "watch?v=" in url:
        return url.split("v=")[1].split("&")[0]

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    if "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]

    return None


def parse_video_list(txt_path: Path):

    dataset = []

    current_jump = "UnknownJump"
    current_rotation = "UnknownRotation"

    with open(txt_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("##"):
                current_rotation = line.lstrip("#").strip()
                continue

            if line.startswith("#"):
                current_jump = line.lstrip("#").strip()
                current_rotation = "UnknownRotation"
                continue

            if "youtube.com" in line or "youtu.be" in line:

                video_id = extract_video_id(line)

                if video_id:
                    dataset.append(
                        {
                            "url": line,
                            "id": video_id,
                            "jump_type": current_jump,
                            "rotation": current_rotation,
                        }
                    )

    return dataset


def main():

    script_dir = Path(__file__).parent

    links_file = script_dir / "video_database.txt"
    data_dir = script_dir.parent / "data"  # project_root/data/

    downloader = YouTubeDownloader(data_dir)

    dataset = parse_video_list(links_file)
    existing_ids = downloader.get_downloaded_ids()

    print("Videos in list:", len(dataset))
    print("Already downloaded:", len(existing_ids))

    for i, item in enumerate(dataset, 1):

        if item["id"] in existing_ids:
            continue

        print(f"\n[{i}/{len(dataset)}]")
        print("Jump:", item["jump_type"])
        print("Rotation:", item["rotation"])
        print("URL:", item["url"])

        metadata = downloader.download(
            item["url"],
            item["jump_type"],
            item["rotation"]
        )

        if metadata:
            existing_ids.add(metadata["video_id"])
            print("Saved:", metadata["title"])
            time.sleep(2)
        else:
            print("Failed")


if __name__ == "__main__":
    main()