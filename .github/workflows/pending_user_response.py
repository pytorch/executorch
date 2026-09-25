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
    issues = repo.get_issues(state="open", labels=[LABEL])
    print(f"[VALIDATION] Would fetch open issues with label '{LABEL}'.")

    now = datetime.datetime.utcnow()

    for issue in issues:
        print(f"[VALIDATION] Would fetch comments for issue/PR #{issue.number}.")
        comments = sorted(
            issue.get_comments(),
            key=lambda comment: comment.created_at,
        )
        
        # Find automation comments
        auto_comments = [
            comment for comment in comments if REMINDER_MARKER in (comment.body or "")
        ]
        latest_auto_comment = auto_comments[-1] if auto_comments else None

        if latest_auto_comment is not None:
            user_responded = any(
                comment.created_at > latest_auto_comment.created_at
                and comment.user is not None
                and comment.user.login == issue.user.login
                and REMINDER_MARKER not in (comment.body or "")
                for comment in comments
            )
            # ---- REMOVE LABEL WHEN USER HAS RESPONDED ----
            if user_responded:
                print(
                    f"User responded to issue/PR #{issue.number}; "
                    f"removing '{LABEL}' label."
                )
                issue.remove_from_labels(LABEL)
                continue
 
            days_since_reminder = (now - latest_auto_comment.created_at).days
            
            # ---- CLOSE ISSUE AFTER 30 DAYS OF REMINDER ----
            if days_since_reminder >= DAYS_BEFORE_CLOSE:
                print(
                    f"Closing issue/PR #{issue.number} due to no response from author."
                )
                issue.create_comment(CLOSE_COMMENT)
                issue.edit(state="closed")
                continue
            # ---- POST REMINDER AFTER 7 DAYS OF INITIAL REMINDER ----
            if days_since_reminder >= REMINDER_COOLDOWN_DAYS:
                print(f"Posting reminder for issue/PR #{issue.number}.")
                issue.create_comment(REMINDER_COMMENT.format(issue.user.login))
            else:
                print(
                    f"Skipping issue/PR #{issue.number}; "
                    "a reminder was posted recently."
                )
            continue

        # ---- INITIAL REMINDER AFTER 30 DAYS OF INACTIVITY ----
        last_comment = comments[-1] if comments else None

        if (
            last_comment is not None
            and (now - last_comment.created_at).days >= DAYS_BEFORE_REMINDER
        ):
            print(f"Posting initial reminder for issue/PR #{issue.number}.")
            issue.create_comment(REMINDER_COMMENT.format(issue.user.login))
        else:
            print(
                f"Skipping issue/PR #{issue.number}; "
                "it has not been inactive for 30 days."
            )


if __name__ == "__main__":
    main()
