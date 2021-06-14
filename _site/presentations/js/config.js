// More info about config & dependencies:
// - https://github.com/hakimel/reveal.js#configuration
// - https://github.com/hakimel/reveal.js#dependencies
Reveal.initialize({
    menu: {
        side: 'left',
        width: 'full',
        numbers: false,
        titleSelector: 'h1, h2, h3, h4, h5, h6',
        useTextContentForMissingTitles: false,
        hideMissingTitles: false,
        markers: true,
        custom: false,
        themes: false,
        themesPath: 'css/theme/sysAI.css',
        transitions: false,
        openButton: true,
        openSlideNumber: false,
        keyboard: true,
        sticky: false,
        autoOpen: true,
        delayInit: false,
        penOnInit: false,
        loadIcons: true
    },
    hash: true,
    controls: true,
    progress: true,
    touch: true,
    keyboard: true,
    help: true,
    transition: "slide",
    transitionSpeed: 'default',
    showNotes: false,
    hideAddressBar: true,
    fragments: true,
    slideNumber: true,
    logo: true,
    width: "90%",
    height: "90%",
    margin: 0,
    minScale: 0.2,
    maxScale: 0.9,
    markdown: {smartypants: true},
    dependencies: [
        { src: 'plugin/markdown/marked.js' },
        { src: 'plugin/markdown/markdown.js' },
        { src: 'plugin/highlight/highlight.js', async: true },
        { src: 'plugin/math/math.js', async: true },
        { src: 'plugin/zoom-js/zoom.js'},
        { src: 'plugin/search/search.js', async: true },
        { src: 'plugin/menu/menu.js'}
    ]
});