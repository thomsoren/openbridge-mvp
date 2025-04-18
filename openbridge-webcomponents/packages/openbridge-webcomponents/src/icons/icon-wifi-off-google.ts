import {LitElement, html, css, svg} from 'lit';
import {customElement, property} from 'lit/decorators.js';

@customElement('obi-wifi-off-google')
export class ObiWifiOffGoogle extends LitElement {
  @property({type: Boolean}) useCssColor = false;

  private icon = svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
<path d="M23.64 7.595C23.19 7.255 18.71 3.595 12 3.595C10.68 3.595 9.44999 3.735 8.30999 3.975L18.43 14.095L23.64 7.595ZM3.40999 1.905L1.99999 3.315L4.04999 5.365C1.90999 6.355 0.589985 7.415 0.359985 7.595L12 22.095L15.91 17.225L19.23 20.545L20.64 19.135L3.40999 1.905Z" fill="currentColor"/>
</svg>
`;

  private iconCss = svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M23.64 7.595C23.19 7.255 18.71 3.595 12 3.595C10.68 3.595 9.44999 3.735 8.30999 3.975L18.43 14.095L23.64 7.595ZM3.40999 1.905L1.99999 3.315L4.04999 5.365C1.90999 6.355 0.589985 7.415 0.359985 7.595L12 22.095L15.91 17.225L19.23 20.545L20.64 19.135L3.40999 1.905Z" style="fill: var(--element-active-color)"/>
</svg>
`;

  override render() {
    return html`
      <div class="wrapper">${this.useCssColor ? this.iconCss : this.icon}</div>
    `;
  }

  static override styles = css`
    .wrapper {
      height: 100%;
      width: 100%;
      line-height: 0;
    }
    .wrapper > * {
      height: 100%;
      width: 100%;
    }
  `;
}

declare global {
  interface HTMLElementTagNameMap {
    'obi-wifi-off-google': ObiWifiOffGoogle;
  }
}
