import os 
from typing import List,Tuple
def loopTowardsTheName(name: str, target_style: List[str]):
    joinContent: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(name):
        for ff in (dirs):
            for _, _, files in os.walk(os.path.join(root,ff)):
                for f in files:
                    for style_path in target_style:
                        joinContent.append( (os.path.join(root, ff, f), style_path ))

    return joinContent

if __name__ == "__main__":
    style_path: List[str] = [
        os.path.join("Pictures", "Styles", "megamendung.png"),
        os.path.join("Pictures", "Styles", "parang.png"),
        os.path.join("Pictures", "Styles", "kawung.png")
    ]
    print(loopTowardsTheName("./Pictures", style_path))