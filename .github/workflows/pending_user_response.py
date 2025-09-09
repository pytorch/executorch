import os
import datetime
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

def main():
    g = Github(os.environ['GH_TOKEN'])
    repo = g.get_repo(REPO_NAME)

    print("[VALIDATION] Would connect to Github and fetch repo:", REPO_NAME)
    issues = repo.get_issues(state='open', labels=[LABEL])
    print(f"[VALIDATION] Would fetch open issues with label '{LABEL}'.")

    now = datetime.datetime.utcnow()

    # Simulate issues for validation workflow
    issues = []  # Replace with mock issues if needed

    for issue in issues:
        # comments = list(issue.get_comments())
        print(f"[VALIDATION] Would fetch comments for issue/PR #{issue.number}.")
        comments = []  # Replace with mock comments if needed
        last_comment = comments[-1] if comments else None

        # Find automation comments
        auto_comments = [c for c in comments if REMINDER_MARKER in c.body]
        user_comments = [c for c in comments if REMINDER_MARKER not in c.body]

        # Case 1: No automation comment yet, and last comment > 30 days ago
        if not auto_comments:
            if last_comment and (now - last_comment.created_at).days >= DAYS_BEFORE_REMINDER:
                # Tag the issue author or PR author
                user = issue.user.login
                # issue.create_comment(REMINDER_COMMENT.format(user))
                print(f"[VALIDATION] Would remind {user} on issue/PR #{issue.number}")

        # Case 2: Automation comment exists, but no user response after 30 more days
        elif auto_comments:
            last_auto = auto_comments[-1]
            # Any user response after automation?
            user_responded = any(
                c.created_at > last_auto.created_at and c.user.login == issue.user.login
                for c in user_comments
            )

            if not user_responded:
                if (now - last_auto.created_at).days >= DAYS_BEFORE_CLOSE:
                    # issue.create_comment(CLOSE_COMMENT)
                    # issue.edit(state="closed")
                    print(f"[VALIDATION] Would close issue/PR #{issue.number} due to inactivity.")

            else:
                # Remove label if user responded
                labels = [l.name for l in issue.labels if l.name != LABEL]
                # issue.set_labels(*labels)
                print(f"[VALIDATION] Would remove label from issue/PR #{issue.number} after user response.")

if __name__ == "__main__":
    main()
