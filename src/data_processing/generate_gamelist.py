from pathlib import Path


def find_games(base_dir_path):
    gamelist = []
    for game in Path(base_dir_path).rglob("*.lud"):
        if "experimental" not in str(game):
            gamepath = str(game).split(base_dir_path)[-1]
            gamelist.append(gamepath)
    return gamelist


if __name__ == "__main__":
    base_dir_path = "data/ludii/expand/"
    gamelist = find_games(base_dir_path)
    with open("data/ludii/gamelist.txt", "w") as f:
        for game in gamelist:
            f.write(f"{game}\n")
    print(f"Created gamelist with {len(gamelist)} games")
