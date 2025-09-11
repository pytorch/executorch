import datetime
import os

from github import Github

REPO_NAME = "pytorch/executorch"
LABEL = "need-user-input"
REMINDER_MARKER = "<!-- executorch-auto-reminder -->"
REMINDER_COMMENT = (
    f"{REMINDER_MARKER}\nHi @{0}, this issue/PR has been marked as 'need-user-input'. "
    "Please respond or provide input. If we don't hear back in 30 days, this will be closed."
)
CLOSE_COMMENT = (
    f"{REMINDER_MARKER}\nClosing due to no response after 30 days. "
    "If you still need help, feel free to re-open or comment again!"
)
DAYS_BEFORE_REMINDER = 30
DAYS_BEFORE_CLOSE = 30
REMINDER_COOLDOWN_DAYS = 7  # Don't post another reminder within 7 days


def main():
    g = Github(os.environ["GH_TOKEN"])
    repo = g.get_repo(REPO_NAME)

    print("[VALIDATION] Would connect to Github and fetch repo:", REPO_NAME)
    issues = repo.get_issues(state='open', labels=[LABEL])
    print(f"[VALIDATION] Would fetch open issues with label '{LABEL}'.")

    now = datetime.datetime.utcnow()

    for issue in issues:
        print(f"[VALIDATION] Would fetch comments for issue/PR #{issue.number}.")
        comments = []  # Replace with mock comments if needed
        last_comment = comments[-1] if comments else None

        # Find automation comments
        auto_comments = [c for c in comments if REMINDER_MARKER in c.body]
        user_comments = [c for c in comments if REMINDER_MARKER not in c.body]

        # ---- REMINDER LOGIC ----
        # Only remind if NO reminder in last 7 days
        recent_auto_reminder = any(
            (now - c.created_at).days < REMINDER_COOLDOWN_DAYS
            for c in auto_comments
        )

        if not auto_comments:
            if (
                last_comment and (now - last_comment.created_at).days >= DAYS_BEFORE_REMINDER
            ):
                user = issue.user.login
                print(
                    f"[VALIDATION] Would remind {user} on issue/PR #{issue.number}"
                )

        elif auto_comments and not recent_auto_reminder:
            # Only post new reminder if last was > REMINDER_COOLDOWN_DAYS ago
            last_auto = auto_comments[-1]
            user = issue.user.login
            if (now - last_auto.created_at).days >= REMINDER_COOLDOWN_DAYS:
                print(
                    f"[VALIDATION] Would remind {user} again on issue/PR #{issue.number}"
                )

        # ---- EXISTING CLOSE/REMOVE LABEL LOGIC ----
        if auto_comments:
            last_auto = auto_comments[-1]
            user_responded = any(
                c.created_at > last_auto.created_at and c.user.login == issue.user.login
                for c in user_comments
            )
            if not user_responded:
                if (now - last_auto.created_at).days >= DAYS_BEFORE_CLOSE:
                    print(
                        f"[VALIDATION] Would close issue/PR #{issue.number} due to inactivity."
                    )
            else:
                print(
                    f"[VALIDATION] Would remove label from issue/PR #{issue.number} after user response."
                )

if __name__ == "__main__":
    main()
