/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const {fbContent} = require('docusaurus-plugin-internaldocs-fb/internal');

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'Executorch',
  tagline: 'A simple and portable executor of PyTorch programs.',
  url: 'https://internalfb.com', // TODO: An external website
  baseUrl: '/',
  onBrokenLinks: 'log',
  onBrokenMarkdownLinks: 'throw',
  trailingSlash: true,
  favicon: 'img/favicon.ico',
  organizationName: 'facebook',
  projectName: 'executorch',

  presets: [
    [
      require.resolve('docusaurus-plugin-internaldocs-fb/docusaurus-preset'),
      {
        docs: {
          // Docs folder path relative to website dir
          path: 'docs',
          // Sidebars file relative to website dir
          sidebarPath: require.resolve('./sidebars.js'),
          // Where to point users when they click "Edit this page"
          editUrl: fbContent({
            internal:
              'https://www.internalfb.com/code/fbsource/fbcode/executorch/docs/website',
            external:
              'https://github.com/pytorch/executorch/',
          }),
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        staticDocsProject: 'executorch',
        trackingFile: 'xplat/staticdocs/WATCHED_FILES',
        'remark-code-snippets': {
          baseDir: '..',
        },
        enableEditor: true,
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Executorch',
        logo: {
          alt: 'Executorch Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'tutorials/setting_up_executorch',
            position: 'left',
            label: 'Documentation',
          },
          ...fbContent({
          internal: [
            {
              label: 'Internal',
              to: 'docs/fb/poc',
              position: 'left',
            },
          ],
          external: [],
          }),
          {
            label: 'API',
            position: 'left',
            items: [
              {
                label: 'Python API',
                to: 'py_api',
                target: 'blank',
              },
              {
                label: 'C++ API',
                to: 'cpp_api',
                target: 'blank',
              },
            ],
          },
          ...fbContent({
          internal: [
            {
              label: 'Code',
              href: 'https://fburl.com/executorch',
              position: 'right',
            },
          ],
          external: [],
          }),
        ],
      },
      footer: {
        style: 'dark',
        copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
});
