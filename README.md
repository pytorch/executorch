# Releasing New Documentation Version

## Adding docs for a new release

Official docs for a release are built and pushed when a final tag
is created via GitHub. When -rc tags are built, you should see a directory
named after the tag with the -rc suffix and the prefix "v"
truncated. For example, v0.1.0-rc1 would become just 0.1. This way
you can preview your documentation before the release.

After the final tag is created, an action will move all the docs from
`tags/vx.x.x` tag to a directory named after your release version.

When the first -rc tag dir for a new release branch is pushed to the gh-pages
branch, you will need to manually update versions.html as follows:

```
<li class="toctree-l1">
  <a class="reference internal" href="0.1/">main (unstable)</a>
</li>
<li class="toctree-l1">
  <a class="reference internal" href="0.1/">v0.2.0 (release candidate)</a> - replace `v0.2.0` with your new tag
</li>
<li class="toctree-l1">
  <a class="reference internal" href="0.1/">v0.1.0 (stable)</a>
</li>
... 
```

### Updating `stable` versions

The stable directory is a symlink to the latest released version.
On the day of the release, you need to update the symlink to the
release version. For example:

```
NEW_VERSION=0.1   # substitute the correct version number here
git checkout gh-pages
rm stable # remove the existing symlink. **Do not** edit!
ln -s "${NEW_VERSION}" stable
git add stable
git commit -m "Update stable to ${NEW_VERSION}"
git push -u origin
```

### Adding version to dropdown

In addition to updating stable, you need to update the dropdown to include
the latest version of docs.

In `versions.html`, rename (release candidate) to (stable), and remove the
old (stable). Here is an example: 

```
<li class="toctree-l1">
  <a class="reference internal" href="0.1/">main (unstable)</a>
</li>
<li class="toctree-l1">
  <a class="reference internal" href="0.2/">v0.2.0 (stable)</a>
</li>
<li class="toctree-l1">
  <a class="reference internal" href="0.1/">v0.1.0</a>
</li>
```

### Adding a <noindex> tag to old versions

You don't want your old documentation to be discoverable by search
engines. Therefore, you can run the following script to add a 
`noindex` tag to all .html files in the old version of the docs.
For exampla, when releasing 0.2, you want to add a noindex tag to all
0.1 documentation. Here is the script:

```
!/bin/bash
# Adds <meta name="robots" content="noindex"> tags to all html files in a
# directory (recursively)
#
# Usage:
# ./add_noindex_tags.sh directory
#
# Example (from the root directory)
# ./scripts/add_no_index_tags.sh docs/1.6.0
if [ "$1" == "" ]; then
  echo "Incorrect usage. Correct Usage: add_no_index_tags.sh <directory>"
  exit 1
fi
find $1 -name "*.html" -print0 | xargs -0 sed -i '/<head>/a \ \ <meta name="robots" content="noindex">'
```

1. Checkout the `gh-pages` branch.
1. Create a new branch out of `gh-pages`.
1. Save the above script into a file called `/tmp/add_noindex_tags.sh`.
1. Run against the old documentation directory. For example:
   ```
   bash /tmp/add_noindex_tags.sh 0.1
   ```
1. Add your changes:
   ```
   git add -A
   ```
1. Submit a PR and merge into the `gh-pages` branch.
