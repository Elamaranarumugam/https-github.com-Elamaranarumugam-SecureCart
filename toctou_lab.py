
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread, Event
from time import perf_counter

checked = Event()
changed = Event()

def victim_application(file_path):
    # TIME OF CHECK
    check_start = perf_counter()

    content = file_path.read_text()

    if content != "SAFE":
        print("Check failed: Resource is not safe.")
        return

    check_time = perf_counter()
    print("1. Check: Resource is SAFE")

    # Pause until the controlled test changes the resource
    checked.set()
    changed.wait()

    # TIME OF USE
    use_start = perf_counter()
    used_content = file_path.read_text()
    use_time = perf_counter()

    print("2. Resource used:", used_content)

    if used_content != "SAFE":
        print("3. TOCTOU vulnerability demonstrated!")
    else:
        print("3. No change detected.")

    print(f"Check duration: {(check_time - check_start) * 1000:.3f} ms")
    print(f"Check-to-use delay: {(use_start - check_time) * 1000:.3f} ms")
    print(f"Use duration: {(use_time - use_start) * 1000:.3f} ms")


def controlled_resource_change(file_path):
    checked.wait()

    # Controlled simulation of a resource changing after the check
    file_path.write_text("UNSAFE")
    print("   Test: Resource changed after the check.")

    changed.set()


def main():
    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "resource.txt"
        file_path.write_text("SAFE")

        print("=== TOCTOU CONTROLLED LAB ===")

        victim = Thread(
            target=victim_application,
            args=(file_path,)
        )

        test = Thread(
            target=controlled_resource_change,
            args=(file_path,)
        )

        victim.start()
        test.start()

        victim.join()
        test.join()


if __name__ == "__main__":
    main()
