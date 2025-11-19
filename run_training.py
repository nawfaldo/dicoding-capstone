import time
import importlib

modules = [
    ("Training Binary Classification Model", "train_binary"),
    ("Training Multiclass Classification Model", "train_multiclass"),
]


def log_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_step(title, module_name):
    log_header(title)

    start = time.time()

    try:
        mod = importlib.import_module(module_name)

        if hasattr(mod, "main"):
            print(f"Running {module_name}...")
            mod.main()

        duration = time.time() - start
        print(f"[OK] Completed: {title}")
        print(f"Time Elapsed: {duration:.2f} seconds\n")

    except Exception as e:
        print("\n[ERROR]")
        print(f"Module       : {module_name}")
        print(f"Description  : {title}")
        print(f"Exception    : {e}\n")
        raise e


def main():
    print("Machine Learning Training Pipeline")
    print("-" * 80)

    for title, module in modules:
        run_step(title, module)

    print("=" * 80)
    print("All steps completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
