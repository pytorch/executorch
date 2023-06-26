/**
 * (c) Meta Platforms, Inc. and affiliates.
 */

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './index.module.css';

const FeatureList = [
  {
    title: <></>,
    // imageUrl: 'img/undraw_docusaurus_react.svg',
    description: (
      <>
        Enable PyTorch to serve new requirements raised by Reality Labs use
        cases while maintaining PyTorch’s ability to serve existing edge use cases.
      </>
    ),
  },
  {
    title: <></>,
    // imageUrl: 'img/undraw_docusaurus_react.svg',
    description: (
      <>
        Provide the best possible UX for end-users running PyTorch models on
        edge devices.
      </>
    ),
  },
  {
    title: <></>,
    // imageUrl: 'img/undraw_docusaurus_react.svg',
    description: (
      <>
        Consistent story + messaging for how this toolchain fits together with
        similar ones targeting specialized server use cases (e.g. server’s
        Static runtime).
      </>
    ),
  },
  {
    title: <></>,
    // imageUrl: 'img/undraw_docusaurus_react.svg',
    description: (
      <>
        Achieve all these with a sustainable architecture that’s aligned across
        all layers of the PyTorch stack.
      </>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--3', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map(({title, description}) => (
            <Feature
              key={title.id}
              title={title}
              description={description}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/tutorials/getting_started">
            Documentation
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="A simple and portable executor of PyTorch programs.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
